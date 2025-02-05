import time
import random
from typing import Iterator
import threading
import os
from textgrid import TextGrid, IntervalTier, Interval
import re

# 假设 agent_main, set_meeting_transcript 等函数已正确导入
from agent.agent import main as agent_main
from agent.agent_tools import set_meeting_transcript
from agent.agent_utils import process_meeting_files

def simulate_real_time_input(text: str, min_delay: float = 0.001, max_delay: float = 0.002) -> Iterator[str]:
    tokens = list(text)
    for token in tokens:
        yield token
        time.sleep(random.uniform(min_delay, max_delay))

def extract_tool_result(llm_response: str) -> str:
    # 使用正则表达式提取“工具结果”部分
    pattern = r"工具结果：(.*)"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        tool_result = match.group(1).strip()
        return tool_result
    else:
        # 如果未找到“工具结果”，则返回完整回复
        return llm_response.strip()

def extract_keywords_from_reasoner_output(llm_response: str) -> str:
    """从reasoner的输出中提取关键词"""
    # 使用正则表达式匹配"关键词："后面的内容
    pattern = r'关键词：(.*?)(?:\n|$)'
    match = re.search(pattern, llm_response)
    if match:
        keywords = match.group(1).strip()
        # 移除可能存在的方括号并分割关键词
        keywords = keywords.replace('[', '').replace(']', '')
        return keywords
    return ""

def main():
    # 设置输入和输出文件夹路径
    input_folder = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed'
    output_folder = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed_response_added'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中所有的 TextGrid 文件
    textgrid_files = [f for f in os.listdir(input_folder) if f.endswith('.TextGrid')]
    
    for textgrid_filename in textgrid_files:
        textgrid_file_path = os.path.join(input_folder, textgrid_filename)
        # 假设对应的 rttm 文件具有相同的前缀名
        rttm_filename = textgrid_filename.replace('.TextGrid', '.rttm')
        rttm_file_path = os.path.join(input_folder, rttm_filename)
        
        if not os.path.exists(rttm_file_path):
            print(f"对应的 rttm 文件未找到：{rttm_file_path}，跳过该文件。")
            continue
        
        print(f"正在处理文件：{textgrid_filename} 和 {rttm_filename}")
        
        # 提取会议记录
        meeting_transcript = process_meeting_files(rttm_file_path, textgrid_file_path)
        
        # 设置会议记录到 agent.py 中
        set_meeting_transcript(meeting_transcript)
        
        # 解析 TextGrid，找到包含“会议助手agent”的间隔及其 xmin 时间
        tg = TextGrid()
        tg.read(textgrid_file_path)
        
        assistant_phrases = []  # 存储包含“会议助手agent”的间隔信息
        for tier in tg.tiers:
            if isinstance(tier, IntervalTier):
                for interval in tier.intervals:
                    if "会议助手agent" in interval.mark:
                        assistant_phrases.append({
                            'tier_name': tier.name,
                            'interval': interval,
                            'xmin': interval.minTime,
                            'xmax': interval.maxTime
                        })
        
        # 模拟会议数据实时输入
        print("模拟会议数据实时输入开始...\n")
        buffer = ""
        agent_triggered = False
        input_question = ""
        assistant_phrase_index = 0  # 用于跟踪当前处理的“会议助手agent”索引
        
        # 存储 LLM 的回复和时间
        agent_responses = []
        cumulative_time_shift = 0.0  # 累积的时间偏移量，用于调整时间
        
        # 使用 simulate_real_time_input 来模拟实时输入
        for token in simulate_real_time_input(meeting_transcript):
            print(token, end='', flush=True)
            buffer += token
            
            # 检测是否包含 "会议助手agent"
            if not agent_triggered and "会议助手agent" in buffer:
                print("\n\n检测到 '会议助手agent'，准备等待完整的问题输入...")
                agent_triggered = True  # 标记已经检测到触发词，等待完整输入
                buffer = buffer.split("会议助手agent", 1)[1].strip()  # 移除触发词前的内容
                
                # 获取当前“会议助手agent”的 xmin 时间
                if assistant_phrase_index < len(assistant_phrases):
                    current_assistant_phrase = assistant_phrases[assistant_phrase_index]
                    assistant_xmin = current_assistant_phrase['xmin']
                    assistant_phrase_index += 1
                else:
                    print("没有更多的 '会议助手agent' 间隔可用。")
                    break  # 跳出循环
                
                # 设置 agent 回复的开始时间为 xmin + 10 秒
                agent_response_start_time = assistant_xmin + 10.0  # 10 秒后
                # 更新累积的时间偏移
                if agent_responses:
                    last_response = agent_responses[-1]
                    if agent_response_start_time <= last_response['end_time']:
                        agent_response_start_time = last_response['end_time'] + 0.1  # 避免重叠，间隔 0.1 秒
            elif agent_triggered:
                # 检测是否输入结束（假设以句号、感叹号或问号结尾）
                if token in ['。', '！', '？']:
                    input_question = buffer.strip()
                    print(f"\n\n已接收到完整的问题：{input_question}\n")
                    
                    # 调用 agent.py 中的 main 函数，并传入 input_question
                    # 捕获 LLM 的回复
                    from io import StringIO
                    import sys
                    
                    # 重定向 stdout 以捕获 LLM 的回复
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    llm_response = agent_main(input_question)
                    
                    # 获取 LLM 的回复
                    llm_response_full = sys.stdout.getvalue()
                    
                    # 重置 stdout
                    sys.stdout = old_stdout
                    
                    print(f"LLM 回复：\n{llm_response_full}")
                    
                    # 提取“工具结果”部分
                    tool_result = extract_tool_result(llm_response_full)
                    
                    # 提取关键词（如果是reasoner的输出）
                    if "行动：信息检索RAG" in llm_response_full:
                        keywords = extract_keywords_from_reasoner_output(llm_response_full)
                        # 更新tool_result，加入关键词信息
                        if keywords:
                            tool_result = f"提取的关键词: {keywords}\n{tool_result}"
                    
                    print(f"提取的工具结果：\n{tool_result}")
                    
                    # 估计 LLM 回复的持续时间
                    avg_speech_rate = 3.0  # 每秒 3 个字符
                    llm_response_length = len(tool_result)
                    llm_response_duration = llm_response_length / avg_speech_rate  # 持续时间（秒）
                    
                    # 计算 agent 回复的结束时间
                    agent_response_end_time = agent_response_start_time + llm_response_duration
                    
                    # 记录 agent 的回复和时间信息
                    agent_responses.append({
                        'start_time': agent_response_start_time,
                        'end_time': agent_response_end_time,
                        'text': tool_result,
                        'duration': llm_response_duration
                    })
                    
                    # 更新累积的时间偏移
                    cumulative_time_shift += llm_response_duration
                    
                    # 重置缓冲区和标志位
                    buffer = ""
                    agent_triggered = False
                    input_question = ""
                    
                else:
                    continue
            
            else:
                continue
        
        print("\n模拟会议数据实时输入结束。")
        
        # 现在，处理 TextGrid 和 rttm 文件，插入 LLM 的回复
        # 更新 tg.maxTime
        tg.maxTime += cumulative_time_shift
        
        # 创建或获取 agent 的 tier
        agent_tier_name = "agent"
        agent_tier = None
        for tier in tg.tiers:
            if tier.name == agent_tier_name:
                agent_tier = tier
                break

        if agent_tier is None:
            # 创建新的 IntervalTier
            agent_tier = IntervalTier(name=agent_tier_name, minTime=0.0, maxTime=tg.maxTime)
            tg.append(agent_tier)
        else:
            agent_tier.maxTime = tg.maxTime

        # 按照 agent_responses 的顺序插入回复，并确保不重叠
        for agent_response in agent_responses:
            # 插入 agent 回复的新间隔
            agent_interval = Interval(minTime=agent_response['start_time'], maxTime=agent_response['end_time'], mark=agent_response['text'])
            agent_tier.addInterval(agent_interval)
        
        # 确保 agent_tier 的 intervals 按 minTime 排序
        agent_tier.intervals.sort(key=lambda x: x.minTime)
        
        # 调整其他 tier 的间隔
        for agent_response in agent_responses:
            # 需要调整的时间段是从 agent_response['start_time'] 到 tg.maxTime
            shift_start_time = agent_response['start_time']
            shift_duration = agent_response['duration']
            for tier in tg.tiers:
                if tier.name != agent_tier_name:
                    new_intervals = []
                    for interval in tier.intervals:
                        # 如果 interval 在 agent_response 之前，保持不变
                        if interval.maxTime <= shift_start_time:
                            new_intervals.append(interval)
                        # 如果 interval 在 agent_response 之后，调整时间
                        elif interval.minTime >= shift_start_time:
                            shifted_interval = Interval(
                                interval.minTime + shift_duration,
                                interval.maxTime + shift_duration,
                                interval.mark
                            )
                            new_intervals.append(shifted_interval)
                        # 如果 interval 跨越 agent_response，需要拆分
                        else:
                            # 前半部分
                            if interval.minTime < shift_start_time:
                                new_interval_before = Interval(
                                    interval.minTime,
                                    shift_start_time,
                                    interval.mark
                                )
                                if new_interval_before.maxTime > new_interval_before.minTime:
                                    new_intervals.append(new_interval_before)
                            # 后半部分
                            if interval.maxTime > shift_start_time:
                                new_interval_after = Interval(
                                    shift_start_time + shift_duration,
                                    interval.maxTime + shift_duration,
                                    interval.mark
                                )
                                if new_interval_after.maxTime > new_interval_after.minTime:
                                    new_intervals.append(new_interval_after)
                    # 更新 tier 的 intervals
                    new_intervals.sort(key=lambda x: x.minTime)
                    tier.intervals = new_intervals
        
        # 更新 agent_tier 的 maxTime
        agent_tier.maxTime = tg.maxTime

        # 保存新的 TextGrid 文件
        output_textgrid_path = os.path.join(output_folder, textgrid_filename)
        tg.write(output_textgrid_path)
        print(f"新的 TextGrid 文件已保存到 {output_textgrid_path}")
    
        # 处理 rttm 文件
        rttm_entries = []
        with open(rttm_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                rttm_entries.append(parts)
        
        # 为每个 agent 回复插入新的条目，并调整后续条目的开始时间
        for agent_response in agent_responses:
            file_name = rttm_entries[0][1] if rttm_entries else 'unknown'
            agent_entry = ['SPEAKER', file_name, '1',
                           f'{agent_response["start_time"]:.3f}',
                           f'{agent_response["duration"]:.3f}',
                           '<NA>', '<NA>', agent_tier_name, '<NA>', '<NA>']
            rttm_entries.append(agent_entry)
            
            new_rttm_entries = []
            for parts in rttm_entries:
                start_time = float(parts[3])
                if start_time > agent_response['start_time']:
                    if parts != agent_entry:
                        start_time += agent_response['duration']
                        parts[3] = f'{start_time:.3f}'
                new_rttm_entries.append(parts)
            
            # 更新 rttm_entries 以便下一个循环使用
            rttm_entries = new_rttm_entries
    
        # 按开始时间排序
        rttm_entries.sort(key=lambda x: float(x[3]))
        
        # 保存新的 rttm 文件
        output_rttm_path = os.path.join(output_folder, rttm_filename)
        with open(output_rttm_path, 'w') as f:
            for parts in rttm_entries:
                line = ' '.join(parts)
                f.write(line + '\n')
        print(f"新的 rttm 文件已保存到 {output_rttm_path}")

if __name__ == "__main__":
    main()
