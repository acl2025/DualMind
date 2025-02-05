import time
import random
from typing import Iterator

def read_meeting_data(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize_simple(text: str) -> list:
    # 将文本按字符分割，包括空格和换行符
    return list(text)

#def simulate_real_time_input(text: str, min_delay: float = 0.01, max_delay: float = 0.2) -> Iterator[str]:
def simulate_real_time_input(text: str, min_delay: float = 0.001, max_delay: float = 0.002) -> Iterator[str]:
    tokens = tokenize_simple(text)
    for token in tokens:
        yield token
        # 模拟真实输入的延迟
        time.sleep(random.uniform(min_delay, max_delay))

def main():
    file_path = '../case_dataset/case_1/case_1.txt'
    meeting_data = read_meeting_data(file_path)
    
    print("模拟会议数据实时输入开始...")
    for token in simulate_real_time_input(meeting_data):
        print(token, flush=True)
        # 这里可以添加将token传递给会议助手agent的逻辑
        # agent.process_token(token)
    
    print("\n模拟会议数据实时输入结束。")

if __name__ == "__main__":
    main()