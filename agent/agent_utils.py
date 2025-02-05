import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field
import re

import time
from threading import Thread

from transformers import TextIteratorStreamer
from typing import List, Dict, Optional, Any, Union, Callable
from textgrid import TextGrid, IntervalTier
from pydantic import BaseModel, Field

import numpy as np

import torch

# 修改部分：提前加载模型
def streaming_callback(token: str):
    print(token, end='', flush=True)


class LocalQwenLLM(LLM):
    model_name: str = Field(..., description="Qwen 模型的名称")
    model: Any = Field(default=None, description="加载的 Qwen 模型")
    tokenizer: Any = Field(default=None, description="加载的 Qwen 分词器或处理器")
    processor: Any = Field(default=None, description="音频模型的处理器")
    sampling_rate: int = Field(default=None, description="音频采样率")  # 添加 sampling_rate 字段
    streaming: bool = Field(default=False, description="是否启用流式输出")
    callback_manager: Optional[CallbackManager] = Field(default=None, description="用于处理回调的管理器")
    streaming_callback: Optional[Callable[[str], None]] = Field(default=None, description="用于流式输出的回调函数")
    is_audio_model: bool = Field(default=False, description="是否是音频模型")

    def __init__(
        self,
        model_name: str,
        streaming: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        is_audio_model: bool = False,
        **kwargs
    ):
        super().__init__(model_name=model_name, is_audio_model=is_audio_model, **kwargs)
        self.streaming = streaming
        self.callback_manager = callback_manager
        self.streaming_callback = streaming_callback

        if self.model_name == "Qwen/Qwen2-Audio-7B-Instruct":
            # 针对音频模型的特殊处理
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
            self.is_audio_model = True
        else:
            # 处理普通文本模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.is_audio_model = False

    def warm_up_model(self):
        warmup_prompt = "你好,请输出一个token单位"
        if self.is_audio_model:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": warmup_prompt}
                ]}
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_length=3,
                    do_sample=False
                )
        else:
            messages = [
                {"role": "system", "content": "你是个有用的助手，严格按照要求解决问题。"},
                {"role": "user", "content": warmup_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False
                )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.is_audio_model:
            # 针对音频模型的推理过程，只接受文本输入
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)

            # 创建一个 TextIteratorStreamer 对象
            streamer = TextIteratorStreamer(self.processor, skip_special_tokens=True)

            # 设置生成参数
            generation_kwargs = dict(
                **inputs,
                max_length=512,
                streamer=streamer,
            )

            # 在单独的线程中运行生成过程
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            start_time = time.time()
            token_count = 0

            for new_text in streamer:
                token_count += 1
                if token_count > 1:
                    generated_text += new_text
                    if self.streaming and self.callback_manager:
                        for handler in self.callback_manager.handlers:
                            if hasattr(handler, 'on_llm_new_token'):
                                handler.on_llm_new_token(token=new_text)
                    elif self.streaming_callback:
                        self.streaming_callback(new_text)

            end_time = time.time()
            if token_count > 0:
                one_token_time = (end_time - start_time) / token_count
                print(f"\nToken count: {token_count}, Total time: {end_time - start_time:.4f} seconds, One token time: {one_token_time:.4f} seconds")
            else:
                print("\nNo tokens generated.")

            thread.join()
            return generated_text
        else:
            # 针对普通文本模型的推理过程
            messages = [
                {"role": "system", "content": "你是个会议助手agent，专门解决会议内容问题。"},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                streamer=streamer,
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            start_time = time.time()
            token_count = 0
            for new_text in streamer:
                print(f"one token time in planner tools: {time.time()}")
                token_count += 1
                generated_text += new_text
                if self.streaming and self.callback_manager:
                    for handler in self.callback_manager.handlers:
                        if hasattr(handler, 'on_llm_new_token'):
                            handler.on_llm_new_token(token=new_text)
                        elif hasattr(handler, 'on_llm_token'):
                            handler.on_llm_token(token=new_text)
                        else:
                            # 如果处理器没有上述方法，可以选择打印或者忽略
                            pass
                elif self.streaming_callback:
                    self.streaming_callback(new_text)

            end_time = time.time()
            if token_count > 0:
                one_token_time = (end_time - start_time) / token_count
                print(f"\nToken count: {token_count}, Total time: {end_time - start_time:.4f} seconds, One token time: {one_token_time:.4f} seconds")
            else:
                print("\nNo tokens generated.")

            thread.join()
            return generated_text
        
    
    def call_with_conversation(self, conversation: List[Dict[str, Any]]):
        if not self.is_audio_model:
            raise ValueError("call_with_conversation is only supported for audio models")

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_array = ele['audio_array']
                        # 确保音频数据为 float32 类型
                        audio_array = audio_array.astype(np.float32)
                        audios.append(audio_array)

        # 打印音频数据的信息
        print(f"Number of audios: {len(audios)}")
        for idx, audio in enumerate(audios):
            print(f"Audio {idx} length: {len(audio)}, dtype: {audio.dtype}, min: {audio.min()}, max: {audio.max()}")

        # 添加 sampling_rate 参数
        inputs = self.processor(
            text=text,
            audios=audios,
            sampling_rate=self.sampling_rate,  # 使用已定义的 sampling_rate
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if self.streaming and self.callback_manager:
                for handler in self.callback_manager.handlers:
                    if hasattr(handler, 'on_llm_new_token'):
                        handler.on_llm_new_token(token=new_text)
            elif self.streaming_callback:
                self.streaming_callback(new_text)

        thread.join()

    @property
    def _llm_type(self) -> str:
        return "local_qwen"


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"行动：{action.tool}\n行动输入：{action.tool_input}\n观察：{observation}\n思考：我现在知道了行动的结果。\n"
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser, BaseModel):
    tool_names: List[str] = Field(default_factory=list, description="可用工具的名称列表")

    def __init__(self, tools: List[Tool], **data):
        super().__init__(**data)
        self.tool_names = [tool.name for tool in tools]

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        action_match = re.search(r"行动：(.*?)\n", llm_output)
        if not action_match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        
        action = action_match.group(1).strip()
        
        if action not in self.tool_names:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        
        input_match = re.search(r"行动输入：(.*)", llm_output, re.DOTALL)
        if not input_match:
            raise ValueError(f"无法解析行动输入: `{llm_output}`")
        
        action_input = input_match.group(1).strip()
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


# 在 agent_utils.py 或单独的实用程序文件中
def parse_rttm_file(rttm_file_path: str) -> List[Dict]:
    time_ranges = []
    with open(rttm_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            _, _, _, start_time, duration, _, _, speaker, _, _ = parts
            start_time = float(start_time)
            duration = float(duration)
            end_time = start_time + duration
            time_ranges.append({'start_time': start_time, 'end_time': end_time, 'speaker': speaker})
    return time_ranges


def parse_textgrid_file(textgrid_file_path: str) -> Dict[str, List[Dict]]:
    tiers = {}
    tg = TextGrid()
    tg.read(textgrid_file_path)

    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            speaker_name = tier.name.strip()
            intervals_list = []
            for interval in tier.intervals:
                xmin = interval.minTime
                xmax = interval.maxTime
                text = interval.mark.strip()
                intervals_list.append({'xmin': xmin, 'xmax': xmax, 'text': text})
            tiers[speaker_name] = intervals_list
    return tiers

def get_text_for_time_range(intervals: List[Dict], start_time: float, end_time: float) -> str:
    text = ''
    for interval in intervals:
        if interval['xmax'] <= start_time:
            continue
        if interval['xmin'] >= end_time:
            break
        text += interval['text']
    return text

def process_meeting_files(rttm_file_path: str, textgrid_file_path: str) -> str:
    time_ranges = parse_rttm_file(rttm_file_path)
    tiers = parse_textgrid_file(textgrid_file_path)
    transcript = ''
    for time_range in time_ranges:
        speaker = time_range['speaker']
        start_time = time_range['start_time']
        end_time = time_range['end_time']
        if speaker in tiers:
            intervals = tiers[speaker]
            text = get_text_for_time_range(intervals, start_time, end_time)
            if text.strip():
                transcript += f"{speaker}：{text}\n"
        else:
            pass  # 如果在 tiers 中找不到对应的 speaker，跳过
    return transcript
