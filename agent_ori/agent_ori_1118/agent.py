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
from agent.agent_tools import get_train_info, summarize_meeting, information_retrieval_rag, set_meeting_transcript , set_meeting_transcript_from_files, \
                            meeting_transcript, tools, current_question, talker_llm, reasoner_llm, classifier_llm # 修改这里
                            
import threading
from queue import Queue
from typing import Any, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

from agent.agent_prompts import talker_prompt, classifier_prompt, reasoner_prompt


talker_llm_chain = LLMChain(llm=talker_llm, prompt=talker_prompt)



classifier_llm_chain = LLMChain(llm=classifier_llm, prompt=classifier_prompt)


reasoner_llm_chain = LLMChain(llm=reasoner_llm, prompt=reasoner_prompt)

def main(input_question: Optional[str] = None):
    global current_question, selected_agent
    if not input_question:
        input_question = "在讨论学校安全问题时，除了食品安全和工地安全，还有哪些方面的安全需要我们特别关注和讨论？"

    current_question = input_question

    classifier_output_queue = Queue()
    talker_output_queue = Queue()
    reasoner_output_queue = Queue()

    classifier_done_event = threading.Event()
    decision_made_event = threading.Event()
    selected_agent = None  # 'talker' 或 'reasoner'

    def run_classifier():
        print("\n运行 classifier...")
        classifier_output = classifier_llm_chain.run(input=current_question)
        classifier_output_queue.put(classifier_output)
        classifier_done_event.set()

    class StreamingCallbackHandler(BaseCallbackHandler):
        def __init__(self, output_queue):
            self.output_queue = output_queue

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.output_queue.put(token)

    def run_talker():
        # 等待 classifier 做出决策
        decision_made_event.wait()
        # 如果 talker 未被选中，停止线程
        if selected_agent != 'talker':
            return
        # 运行 LLMChain
        callback_handler = StreamingCallbackHandler(talker_output_queue)
        callback_manager = CallbackManager([callback_handler])
        talker_llm.callback_manager = callback_manager
        talker_llm.streaming = True
        talker_llm_chain.run(input=current_question)
        # 生成完成的标志
        talker_output_queue.put(None)

    def run_reasoner():
        # 等待 classifier 做出决策
        decision_made_event.wait()
        if selected_agent != 'reasoner':
            return
        # 首先运行 reasoner，获取初始输出
        full_output = reasoner_llm_chain.run(input=current_question, intermediate_steps=[])
        # 解析输出
        output_parser = CustomOutputParser(tools)
        parsed_output = output_parser.parse(full_output)
        if isinstance(parsed_output, AgentAction):
            action_name = parsed_output.tool
            action_input = parsed_output.tool_input
            tool = next((tool for tool in tools if tool.name == action_name), None)
            if tool is None:
                print(f"未知的工具：{action_name}")
                return
            else:
                print(f"\n使用工具：{action_name}")
                tool_result = tool.func(action_input)
                print(f"\n工具结果：{tool_result}")
                # 生成完成的标志
                reasoner_output_queue.put(None)
        elif isinstance(parsed_output, AgentFinish):
            # 最终答案，直接流式输出
            final_answer = parsed_output.return_values['output']
            for token in final_answer:
                reasoner_output_queue.put(token)
            reasoner_output_queue.put(None)
        else:
            print("无法解析 Reasoner 的输出")

    # 启动线程
    classifier_thread = threading.Thread(target=run_classifier)
    talker_thread = threading.Thread(target=run_talker)
    reasoner_thread = threading.Thread(target=run_reasoner)

    classifier_thread.start()
    talker_thread.start()
    reasoner_thread.start()

    # 等待 classifier 完成
    classifier_done_event.wait()

    # 获取 classifier 的输出
    classifier_output = classifier_output_queue.get()
    print(f"\nclassifier的输出：{classifier_output}")

    # 决定使用哪个智能体
    if "1" in classifier_output:
        selected_agent = 'talker'
    else:
        selected_agent = 'reasoner'

    print(f"\n选择的智能体：{selected_agent}")

    # 通知已做出决策
    decision_made_event.set()

    # 开始输出选定智能体的 Token
    if selected_agent == 'talker':
        output_queue = talker_output_queue
    else:
        output_queue = reasoner_output_queue

    print("\n选定智能体的输出：")
    while True:
        token = output_queue.get()
        if token is None:
            break
        print(token, end='', flush=True)

    print("\n生成完成。")

    # 等待已选择的智能体线程完成
    if selected_agent == 'talker':
        talker_thread.join()
    else:
        reasoner_thread.join()

if __name__ == "__main__":
    main()