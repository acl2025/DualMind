import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field
import re

from web_crawler.web_crawler import NewsByTransfer
import time
from transformers import TextIteratorStreamer
from threading import Thread

from agent.agent_utils import LocalQwenLLM, CustomPromptTemplate, CustomOutputParser, process_meeting_files,streaming_callback
from langchain.chains import LLMChain

# 新增：用于存储会议记录的全局变量
meeting_transcript = ""
tools = None
current_question = None

# 全局模型变量
global talker_llm, classifier_llm, reasoner_llm
talker_llm = None
classifier_llm = None
reasoner_llm = None

def initialize_models(model_name=None):
    # if model_name == 'talker':
    #     print("Loading talker_llm...")
    #     talker_llm = LocalQwenLLM(model_name="Qwen/Qwen2-Audio-7B-Instruct", streaming_callback=streaming_callback, streaming=True, is_audio_model=True,)
    #     return talker_llm
    if model_name == 'classifier':
        print("Loading classifier_llm...")
        classifier_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback, streaming=True, is_audio_model=False)
        return classifier_llm
    elif model_name == 'reasoner':
        print("Loading reasoner_llm...")
        reasoner_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback, streaming=True, is_audio_model=False)
        return reasoner_llm
    else:
        # 如果未指定模型名称，则加载所有模型并返回字典
        print("Loading all models...")
        #talker_llm = LocalQwenLLM(model_name="Qwen/Qwen2-Audio-7B-Instruct", streaming_callback=streaming_callback)
        classifier_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback)
        reasoner_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback)
        return {
            #'talker': talker_llm,
            'classifier': classifier_llm,
            'reasoner': reasoner_llm
        }

def append_meeting_transcript(text: str):
    global meeting_transcript
    meeting_transcript += text

def set_meeting_transcript(transcript: str):
    global meeting_transcript
    meeting_transcript = transcript


def set_meeting_transcript_from_files(rttm_file_path: str, textgrid_file_path: str):
    global meeting_transcript
    meeting_transcript = process_meeting_files(rttm_file_path, textgrid_file_path)



def get_train_info(query: str) -> str:
    match = re.search(r'查询日期为(\d{4}-\d{2}-\d{2}).*?出发地为([\u4e00-\u9fa5]+).*?目的地为([\u4e00-\u9fa5]+)', query)
    if match:
        date, from_station, to_station = match.groups()
        crawler = NewsByTransfer(from_station, to_station, date)
        js_url = crawler.getOneJsUrl()
        csv_list = crawler.getOneNews(js_url)
        if csv_list:
            return str(csv_list)
        else:
            return "未能获取数据，请检查网络连接或网站结构是否发生变化"
    else:
        return "无法从查询中提取日期和站点信息，请确保输入格式正确"

# 新增：会议内容总结工具
def summarize_meeting(reasoner_llm, meeting_transcript) -> str:

    transcript_text = meeting_transcript.text
    if transcript_text:
        # 使用已加载的 llm 进行会议总结
        def streaming_callback(token: str):
            print(token, end='', flush=True)
        prompt = f"请根据以下会议记录内容，生成一份会议总结：\n\n{transcript_text}\n\n会议总结："
        summary = reasoner_llm(prompt)
        return summary
    else:
        return "当前没有可用的会议记录内容。"


def information_retrieval_rag(action_input: str, reasoner_llm, meeting_transcript) -> str:
    # 从 meeting_transcript 中获取文本
    transcript_text = meeting_transcript.text
    
    # 清理和提取关键词
    def extract_keywords(input_text: str) -> list:
        # 移除"关键词："前缀如果存在
        if "关键词：" in input_text:
            input_text = input_text.split("关键词：")[1]
        
        # 分词并清理
        keywords = [
            word.strip() 
            for word in input_text.split() 
            if word.strip() and not word.strip() in ['的', '了', '和', '与', '及', '或']
        ]
        
        return keywords
    
    # 计算两个字符串的相似度
    def string_similarity(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0
        shorter = s1 if len(s1) <= len(s2) else s2
        longer = s2 if len(s1) <= len(s2) else s1
        matching_chars = sum(1 for char in shorter if char in longer)
        return matching_chars / len(shorter)
    
    # 新增：根据位置提取上下文窗口
    def extract_context_window(text: str, position: int, window_size: int = 300) -> str:
        # 获取文本总长度
        text_length = len(text)
        
        # 计算窗口起始和结束位置
        start = max(0, position - window_size)
        end = min(text_length, position + window_size)
        
        # 调整到最近的完整句子
        if start > 0:
            # 向前找到句子开始（句号、问号、感叹号后的位置）
            while start > 0 and text[start] not in '。！？\n':
                start -= 1
            start += 1  # 移到标点符号后的字符
            
        if end < text_length:
            # 向后找到句子结束
            while end < text_length and text[end] not in '。！？\n':
                end += 1
            if end < text_length:
                end += 1  # 包含句号
        
        # 提取上下文并添加标记
        context = text[start:end]
        
        # 在关键词匹配位置添加标记
        relative_pos = position - start
        if 0 <= relative_pos < len(context):
            context = context[:relative_pos] + "【" + context[relative_pos:relative_pos+1] + "】" + context[relative_pos+1:]
        
        return context, (start, end)
    
    # 查找所有匹配位置
    def find_all_matches(text: str, keywords: list, similarity_threshold: float) -> list:
        matches = []
        
        for keyword in keywords:
            window_size = len(keyword)
            # Use transcript_text instead of meeting_transcript
            for i in range(len(transcript_text) - window_size + 1):
                phrase = transcript_text[i:i + window_size]
                similarity = string_similarity(keyword, phrase)
                
                if similarity >= similarity_threshold:
                    matches.append((i, similarity, keyword))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    # 合并重叠的上下文窗口
    def merge_overlapping_contexts(contexts: list) -> list:
        if not contexts:
            return []
        
        # 按起始位置排序
        contexts.sort(key=lambda x: x[1][0])
        
        merged = []
        current_text = contexts[0][0]
        current_start = contexts[0][1][0]
        current_end = contexts[0][1][1]
        
        for text, (start, end) in contexts[1:]:
            if start <= current_end:
                # 有重叠，合并窗口
                current_end = max(current_end, end)
                current_text = text[:start-current_start] + current_text[start-current_start:end-start]
            else:
                # 无重叠，添加当前窗口并开始新的窗口
                merged.append((current_text, (current_start, current_end)))
                current_text = text
                current_start = start
                current_end = end
        
        merged.append((current_text, (current_start, current_end)))
        return merged
    
    # 主处理逻辑
    keywords = extract_keywords(action_input)
    print(f"\n提取的关键词: {keywords}")
    
    # 第一轮搜索（高阈值）
    SIMILARITY_THRESHOLD = 0.6
    matches = find_all_matches(meeting_transcript, keywords, SIMILARITY_THRESHOLD)
    
    # 如果没找到匹配，进行第二轮搜索（低阈值）
    if not matches:
        print("\n未找到高相似度匹配，尝试降低匹配阈值...")
        SIMILARITY_THRESHOLD = 0.4
        matches = find_all_matches(transcript_text, keywords, SIMILARITY_THRESHOLD) 
    
    # 获取所有匹配位置的上下文窗口
    context_windows = []
    for position, similarity, keyword in matches[:10]:
        context, bounds = extract_context_window(transcript_text, position)
        context_windows.append((context, bounds))
    
    # 合并重叠的上下文窗口
    merged_contexts = merge_overlapping_contexts(context_windows)
    
    # 构建最终上下文
    if merged_contexts:
        contexts = [text for text, _ in merged_contexts]
        context = "\n---\n".join(contexts)
    else:
        context = "未找到相关信息"
    
    print(f"\n找到的上下文片段数量: {len(merged_contexts)}")
    
    # **新增**：将 RAG used context 加入返回结果中
    result = f"RAG used context: {context}\n"

    # 构建 prompt
    prompt = f"""根据以下会议内容回答问题，回复字数一定在100字以内，且绝对不能输出分行号（例如"\n"）：
### 问题 ###
{current_question}
###

### 会议内容 ###
{context}
###

请根据以上会议内容用150字以内回答问题。不要回复无关内容。
所提供的会议内容中，【】内的文字为关键词匹配位置。请特别关注这些位置的相关信息。
"""

    # 使用 LLM 生成答案
    answer = reasoner_llm(prompt)
    result += answer
    return result






#定义agent工具
tools = [
        Tool(
            name="高铁价格查询",
            func=get_train_info,
            description="用于获取指定日期和路线的高铁价格信息"
        ),
        Tool(
            name="会议总结工具",
            func=lambda action_input: summarize_meeting(reasoner_llm, meeting_transcript),
            description="提供会议总结并用于对当前的会议内容进行总结，遇到会议总结要求的时候可以直接使用"
        ),
        Tool(
            name="信息检索RAG",
            func=lambda action_input: information_retrieval_rag(action_input, reasoner_llm, meeting_transcript),
            description=(
                "利用实时记录的会议信息，根据问题提供关键词作为行动输入，然后根据该关键词在实时记录的会议信息"
                "遇到需要从会议记录中检索信息的问题时使用此工具。行动输入应包含关键词和问题,关键词越短越好。"
                "只在需要用到会议之前的信息的时候才选择使用此工具,针对问题时不需要使用该工具。"
            )
        )
    ]

# 添加新的工具到工具列表，给reasoner进程的专属函数
def get_tools(reasoner_llm=None, meeting_transcript=None):
    return [
        Tool(
            name="高铁价格查询",
            func=get_train_info,
            description="用于获取指定日期和路线的高铁价格信息"
        ),
        Tool(
            name="会议总结工具",
            func=lambda action_input: summarize_meeting(reasoner_llm, meeting_transcript),
            description="提供会议总结并用于对当前的会议内容进行总结，遇到会议总结要求的时候可以直接使用"
        ),
        Tool(
            name="信息检索RAG",
            func=lambda action_input: information_retrieval_rag(action_input, reasoner_llm, meeting_transcript),
            description=(
                "利用实时记录的会议信息，根据问题提供关键词作为行动输入，然后根据该关键词在实时记录的会议信息"
                "遇到需要从会议记录中检索信息的问题时使用此工具。行动输入应包含关键词和问题,关键词越短越好。"
                "只在需要用到会议之前的信息的时候才选择使用此工具,针对问题时不需要使用该工具。"
            )
        )
    ]




# 预先加载两个模型
#talker_llm = LocalQwenLLM(model_name="opencompass/CompassJudger-1-7B-Instruct", streaming_callback=streaming_callback)
#talker_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback)

# talker_llm = LocalQwenLLM(model_name="Qwen/Qwen2-Audio-7B-Instruct", streaming_callback=streaming_callback)
# classifier_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback)
# reasoner_llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", streaming_callback=streaming_callback)


