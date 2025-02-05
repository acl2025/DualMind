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

from agent.agent_utils import LocalQwenLLM, CustomPromptTemplate, CustomOutputParser, process_meeting_files
from agent.agent_tools import (
    get_train_info, summarize_meeting, information_retrieval_rag, set_meeting_transcript,
    set_meeting_transcript_from_files, meeting_transcript, tools, current_question,
    talker_llm, reasoner_llm, classifier_llm,# 直接引用全局模型实例
)
                            
import threading
from queue import Queue
from typing import Any, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager


import multiprocessing
import time
from queue import Empty

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from threading import Thread
import torch

# 定义 StreamingCallbackHandler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.output_queue.put(token)

# 定义 classifier 进程函数
def classifier_process_func(input_queue, output_queue):
    from agent.agent_tools import initialize_models
    from agent.agent_prompts import classifier_prompt
    # 在子进程中初始化模型，并获取实例
    classifier_llm = initialize_models('classifier')
    classifier_llm_chain = LLMChain(llm=classifier_llm, prompt=classifier_prompt)
    
    while True:
        input_question = input_queue.get()
        if input_question == 'STOP':
            break

        print("\n运行 classifier...")
        classifier_output = classifier_llm_chain.run(input=input_question)
        output_queue.put(classifier_output)

# 定义 talker 进程函数
def talker_process_func(input_queue, output_queue):


    print("Initializing talker model in talker_process_func...")

    # Initialize model and processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct") 
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto",torch_dtype="auto", ) 

    # model.eval()
    
    output_result_flag = False

    while True:
        input_data = input_queue.get()
        if input_data == 'STOP':
            break
        

        # Expect input_data to be a dictionary containing 'audio' and 'text'
        audio_array = input_data['audio']
        text_instruction = input_data['text']

        # Ensure audio_array is a numpy array of float32
        import numpy as np
        audio_array = np.array(audio_array, dtype=np.float32)

        # Check if audio is mono; if not, convert to mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Resample audio to 16000 Hz if necessary
        sampling_rate = 16000
        if input_data.get('sr', 16000) != sampling_rate:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=input_data['sr'], target_sr=sampling_rate)

        # Build conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_instruction},
                    {"type": "audio", "audio_url": "local_audio_1"}
                ]
            }
        ]

        # Prepare text input
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Build audios list, match audio_url
        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        if ele['audio_url'] == 'local_audio_1':
                            audios.append(audio_array)
                        else:
                            pass  # Handle other audio inputs if necessary

        # Create model input
        inputs = processor(
            text=text,
            audios=audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device
        device = model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Create a streamer
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

        # Set generation parameters
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1
        )

        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        start_time = time.time()
        thread.start()

        # 从 streamer 中获取输出，并通过队列发送回主进程
        token_count = 0
        for new_text in streamer:
            if new_text.strip():
                print(f"talker first token time: {time.time()}")
                output_queue.put(new_text)
                token_count += 1

        output_queue.put('__end__')
        end_time = time.time()
        total_time = end_time - start_time
        one_token_time = total_time / token_count if token_count > 0 else 0
        # 将计时信息也通过队列发送回主进程
        time_info = f"\nToken count: {token_count}, Total time: {total_time:.4f} seconds, One token time: {one_token_time:.4f} seconds"
        output_queue.put(time_info)

# 定义 reasoner 进程函数
def reasoner_process_func(input_queue, output_queue, meeting_transcript):
    from agent.agent_tools import initialize_models, get_tools
    from agent.agent_utils import CustomOutputParser
    from langchain.chains import LLMChain
    from agent.agent_prompts import reasoner_prompt
    from queue import Empty
    import time

    # 在子进程中初始化模型，并获取实例
    reasoner_llm = initialize_models('reasoner')

    # 获取工具列表，传入 reasoner_llm 和 meeting_transcript
    tools = get_tools(reasoner_llm, meeting_transcript)
    reasoner_llm_chain = LLMChain(llm=reasoner_llm, prompt=reasoner_prompt)

    while True:
        input_question = input_queue.get()
        if input_question == 'STOP':
            break
        if input_question == 'CANCEL':
            continue

        # 运行 reasoner
        full_output = reasoner_llm_chain.run(input=input_question, intermediate_steps=[])
        output_queue.put(full_output)  # 将 reasoner 的输出发送回主进程

        # 解析输出
        output_parser = CustomOutputParser(tools)
        parsed_output = output_parser.parse(full_output)

        if isinstance(parsed_output, AgentAction):
            action_name = parsed_output.tool
            action_input = parsed_output.tool_input
            tool = next((tool for tool in tools if tool.name == action_name), None)
            if tool is None:
                output_queue.put(f"未知的工具：{action_name}")
                output_queue.put('__end__')
                continue
            else:
                output_queue.put(f"\n使用工具：{action_name}")
                tool_result = tool.func(action_input)
                output_queue.put(f"\nagent工具结果：{tool_result}")
                output_queue.put('__end__')
        elif isinstance(parsed_output, AgentFinish):
            final_answer = parsed_output.return_values['output']
            output_queue.put(final_answer)
            output_queue.put('__end__')
        else:
            output_queue.put("无法解析 Reasoner 的输出")
            output_queue.put('__end__')



def planner_process_func(input_queue, output_queue, meeting_transcript):
    import time
    from threading import Thread
    from queue import Empty
    from agent.agent_utils import LocalQwenLLM, CustomOutputParser
    from agent.agent_tools import get_tools
    from transformers import TextIteratorStreamer

    # 初始化 planner_llm，启用流式输出
    planner_llm = LocalQwenLLM(
        model_name="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        streaming=True,
        streaming_callback=None  # 在函数内部处理流式输出
    )

    # 执行预热
    planner_llm.warm_up_model()

    # 获取工具列表，传入 planner_llm 和 meeting_transcript
    tools = get_tools(planner_llm, meeting_transcript)
    output_parser = CustomOutputParser(tools)

    while True:
        try:
            input_question = input_queue.get(timeout=0.1)
        except Empty:
            continue

        if input_question == 'STOP':
            break
        if input_question == 'CANCEL':
            continue

        # 构造 prompt
        prompt = f"""
        ### 要求 ###
        首先你需要判断以下问题是否是简短问题：
        - 如果问题超过90字或问题中包含"基于之前"等类似字样，输出："0"。输出完"0"后必须根据示例输出,你只能选择使用信息检索RAG。
        - 如果问题少于90字或问题中不包含"基于之前"等类似字样，只输出："1"。输出完"1"后则不能继续输出。
        
        ###

        ### 工具 ###
        [
        Tool(
        name="信息检索RAG",
        func=lambda action_input: information_retrieval_rag(action_input, planner_llm, meeting_transcript),
        description=(
        "利用实时记录的会议信息，根据问题提供关键词作为行动输入，然后根据该关键词在实时记录的会议信息"
        "遇到需要从会议记录中检索信息的问题时使用此工具。行动输入应包含关键词和问题,关键词越短越好。"
        "只在需要用到会议之前的信息的时候才选择使用此工具,针对问题时不需要使用该工具。"
        )
        )
        ]
        ###
        
        ### 示例问题1 ###
        基於之前我們討論的內容 關於新開業攝影樓的宣傳和營銷策略你怎麼看待我們提出的前三天打五折活動 以及定期抽選幸運顧客免費拍攝的方案再稍微算Time片這些活動是否能有效提升我們的知名度和吸引新客戶同时我们应该如何平衡优惠活动和成本控制以确保公司的长期盈利
        ###

        ### 示例输出1 ###
        0
        1. 行动：信息检索RAG
        2. 行动输入：关键词：免費拍攝 打五折活動 优惠活动 吸引新客戶
        ###
        
        ### 示例问题2 ###
        如何通过线上渠道提高库存服装的销量并预测市场趋势然后我觉得还有一点就是调查
        ###

        ### 示例输出2 ###
        1
        ###

        ### 你需要判断的问题###
        {input_question}
        ###
    """
        #print(f"Prompt:\n{prompt}\n")

        # 准备模型输入
        messages = [
            {"role": "system", "content": "你是会议助手agent，帮助处理会议内容信息。"},
            {"role": "user", "content": prompt}
        ]
        text = planner_llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = planner_llm.tokenizer([text], return_tensors="pt").to(planner_llm.model.device)

        # 创建一个 TextIteratorStreamer
        streamer = TextIteratorStreamer(
            tokenizer=planner_llm.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # 设置生成参数
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer
        )

        # 定义生成过程
        def generate():
            planner_llm.model.generate(**generation_kwargs)

        # 启动生成线程
        generator_thread = Thread(target=generate)
        generator_thread.start()

        # 流式处理输出，检测第一个 token
        first_token = None
        collected_output = ""
        for new_token in streamer:
            print(f"planner new_token time: {time.time()}")
            if new_token == '':
                continue
            output_queue.put(new_token)
            collected_output += new_token

            if first_token is None:
                # 第一个 token
                first_token = new_token.strip()
                print(f"First token received: {first_token}, time: {time.time()}")

                if first_token == '1':
                    print("判断是简短问题，planner结束输出, time: {time.time()}")
                    # 简短问题，结束
                    output_queue.put('__end__')
                    # 停止生成
                    break
                elif first_token == '0':
                    print("===planner 进一步输出开始===, time: {time.time()}")
                    # 复杂问题，继续处理
                    # 在这里继续收集输出
                    pass
                else:
                    # 无法识别的 token
                    output_queue.put("planner_process_func未能正确解析到 0/1，原始判定输出：")
                    output_queue.put(collected_output)
                    output_queue.put('__end__')
                    # 停止生成
                    break
            else:
                # 已经接收到第一个 token，继续收集剩余输出
                pass

        # 等待生成线程结束
        generator_thread.join()

        # 如果是复杂问题，处理 agent 功能
        if first_token == '0':
            # 使用收集的输出，解析 agent 行动
            remaining_output = collected_output[1:].strip()  # 去除第一个 token '0'
            print(f"Remaining output after first token:\n{remaining_output}\n")

            # 解析输出
            parsed_output = output_parser.parse(remaining_output)

            if isinstance(parsed_output, AgentAction):
                action_name = parsed_output.tool
                action_input = parsed_output.tool_input
                tool = next((tool for tool in tools if tool.name == action_name), None)
                if tool is None:
                    output_queue.put(f"未知的工具：{action_name}")
                    output_queue.put('__end__')
                    continue
                else:
                    output_queue.put(f"\n使用工具：{action_name}")
                    tool_result = tool.func(action_input)
                    output_queue.put(f"\nagent工具结果：{tool_result}")
                    output_queue.put('__end__')
            elif isinstance(parsed_output, AgentFinish):
                final_answer = parsed_output.return_values['output']
                output_queue.put(final_answer)
                output_queue.put('__end__')
            else:
                output_queue.put("无法解析 Planner 的输出")
                output_queue.put('__end__')
        elif first_token == '1':
            # 简短问题已处理，直接结束
            pass
        else:
            # 无法识别的 token
            pass
