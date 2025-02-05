import time
import random
from typing import Iterator
from agent.agent import main as agent_main
from agent.agent_tools import set_meeting_transcript, set_meeting_transcript_from_files
from meeting_simulator.meeting_simulator import read_meeting_data, tokenize_simple, simulate_real_time_input
import threading

from agent.agent_utils import process_meeting_files


def simulate_real_time_input(text: str, min_delay: float = 0.001, max_delay: float = 0.002) -> Iterator[str]:
    # 将文本按字符分割，包括空格和换行符
    tokens = list(text)
    for token in tokens:
        yield token
        # 模拟真实输入的延迟
        #print(token, end=' ', flush=True)
        time.sleep(random.uniform(min_delay, max_delay))

def main():
    # 设置文件路径
    # 以下文件成功实现了一次talker直接回答
    # rttm_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/L_R003S01C02_agent_added_fixed.rttm'
    # textgrid_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/L_R003S01C02_agent_added_fixed.TextGrid'
    
    # 以下文件很成功
    # rttm_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/L_R004S01C01_agent_added_fixed.rttm'
    # textgrid_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/L_R004S01C01_agent_added_fixed.TextGrid'
    
    
    rttm_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/S_R004S04C01_agent_added_fixed.rttm'
    textgrid_file_path = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed/S_R004S04C01_agent_added_fixed.TextGrid'
    
    print(rttm_file_path)
    
    # 提取会议记录
    meeting_transcript = process_meeting_files(rttm_file_path, textgrid_file_path)
    
    # 设置会议记录到 agent.py 中
    set_meeting_transcript(meeting_transcript)
    
    # 模拟会议数据实时输入
    print("模拟会议数据实时输入开始...\n")
    buffer = ""
    agent_triggered = False
    input_question = ""
    
    skip_tokens_count = 0  # 添加计数器来跟踪需要跳过的token数量
    TOKENS_TO_SKIP = 70    # 设置需要跳过的token数量
    
    # 使用 simulate_real_time_input 来模拟实时输入
    
    # 避免连续两次检测到同一个agent trigger， 可进一步优化，token数为agent trigger那句话的token数
    for token in simulate_real_time_input(meeting_transcript):
        if skip_tokens_count > 0:
            skip_tokens_count -= 1
            continue
        
        print(token, end='', flush=True)
        buffer += token
        
        # 检测是否包含 "会议助手agent"
        if "会议助手agent" in buffer and not agent_triggered:
            print("\n\n检测到 '会议助手agent'，准备等待完整的问题输入...")
            agent_triggered = True  # 标记已经检测到触发词，等待完整输入
            buffer = buffer.split("会议助手agent", 1)[1].strip()  # 移除触发词前的内容
            
        elif agent_triggered:
            # 检测是否输入结束（假设以句号、感叹号或问号结尾）
            if token in ['。', '！', '？']:
                input_question = buffer.strip()
                print(f"\n\n已接收到完整的问题：{input_question}\n")
                
                # 调用 agent.py 中的 main 函数，并传入 input_question
                agent_thread = threading.Thread(target=agent_main, args=(input_question,))
                agent_thread.start()
                agent_thread.join()
                
                # 重置缓冲区和标志位
                buffer = ""
                agent_triggered = False
                input_question = ""
                
                skip_tokens_count = TOKENS_TO_SKIP 
                
                #避免连续两次检测到同一个agent trigger
                ## 你需要写的代码
                    
        else:
            # 正常累积 buffer
            continue
    
    print("\n模拟会议数据实时输入结束。")


if __name__ == "__main__":
    main()
