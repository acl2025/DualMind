import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Optional, Union, Callable
from threading import Thread
import time
import torch
import sys
import queue

from agent.agent_utils import (
    LocalQwenLLM, CustomPromptTemplate, CustomOutputParser,
    process_meeting_files, streaming_callback
)
from agent.agent_tools import (
    initialize_models, get_tools, set_meeting_transcript_from_files,
    meeting_transcript, current_question
)
from agent.agent_prompts import planner_prompt

# 定义 planner 进程函数
def planner_process_func(input_queue, output_queue, meeting_transcript):
    from agent.agent_utils import LocalQwenLLM
    from agent.agent_tools import get_tools
    from agent.agent_prompts import planner_prompt

    # 在子进程中初始化模型，并获取实例
    print("Loading planner_llm...")
    planner_llm = LocalQwenLLM(
        model_name="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        streaming=True,
        is_audio_model=False
    )

    # 预热模型
    def warm_up_model(model, tokenizer):
        warmup_prompt = "你好"
        warmup_inputs = tokenizer([warmup_prompt], return_tensors="pt").to(model.device)
        print("=========== Warmup: Start ===========")
        with torch.no_grad():
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=1,
                do_sample=False
            )
        print("=========== Warmup: End ===========\n")

    warm_up_model(planner_llm.model, planner_llm.tokenizer)

    tools = get_tools(planner_llm, meeting_transcript)

    while True:
        input_question = input_queue.get()
        if input_question == 'STOP':
            break

        # 使用 planner_prompt 构建输入
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": planner_prompt.format(input=input_question)}
        ]

        text = planner_llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = planner_llm.tokenizer([text], return_tensors="pt").to(planner_llm.model.device)

        # 第一次生成，判断 0/1
        first_generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            top_p=0.9,
            temperature=0.7
        )

        streamer = planner_llm.streamer

        def generate_thread_func():
            with torch.no_grad():
                planner_llm.model.generate(**first_generation_kwargs, streamer=streamer)

        gen_thread = Thread(target=generate_thread_func)
        gen_thread.start()

        first_gen_text = ""
        for new_text in streamer:
            print(f"planner first judgetoken time: {time.time()}")
            first_gen_text += new_text
            output_queue.put(new_text)

        gen_thread.join()

        # 解析生成的第一个 token
        original_input_text = planner_llm.tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
        index = first_gen_text.find(original_input_text)
        if index != -1:
            new_gen_part = first_gen_text[index + len(original_input_text):]
        else:
            new_gen_part = first_gen_text

        judge_str = new_gen_part.strip()
        judge_token = judge_str[-1] if len(judge_str) > 0 else ""

        if judge_token == "1":
            output_queue.put('__end__')
            continue  # 简单问题，不需要后续生成

        elif judge_token == "0":
            # 第二次生成，继续流式输出
            second_generation_kwargs = dict(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

            streamer = planner_llm.streamer

            def second_generate_thread_func():
                with torch.no_grad():
                    planner_llm.model.generate(**second_generation_kwargs, streamer=streamer)

            gen_thread = Thread(target=second_generate_thread_func)
            gen_thread.start()

            for new_text in streamer:
                print(f"planner second token time: {time.time()}")
                output_queue.put(new_text)

            gen_thread.join()
            output_queue.put('__end__')

        else:
            output_queue.put("无法解析 Planner 的输出")
            output_queue.put('__end__')
            continue