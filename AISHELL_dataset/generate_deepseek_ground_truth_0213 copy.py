import os
import json
from tqdm import tqdm
import openai
import tiktoken

# 设置OpenAI API密钥
openai.api_key = 'key'

# 定义常量
INPUT_JSONL_DIR = '/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103'
OUTPUT_DIR = '/home/leon/agent/AISHELL_dataset/train_S/ground_truth_train_S'
PROCESSED_RECORD = os.path.join(OUTPUT_DIR, '.processed_files')

# 初始化tiktoken编码器
encoding = tiktoken.encoding_for_model("gpt-4-0125-preview")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_files():
    """加载已处理的文件记录"""
    if os.path.exists(PROCESSED_RECORD):
        with open(PROCESSED_RECORD, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    return set()

def save_processed_file(file_name):
    """保存已处理的文件记录"""
    with open(PROCESSED_RECORD, 'a', encoding='utf-8') as f:
        f.write(file_name + '\n')

def count_tokens(text):
    """计算文本的token数量"""
    return len(encoding.encode(text))

def ask_gpt(prompt):
    """调用GPT API获取回答"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "你是一个帮助回答问题的助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        raise Exception(f"GPT API error: {str(e)}")

def process_jsonl_file(jsonl_path):
    """处理单个jsonl文件"""
    file_name = os.path.basename(jsonl_path)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    # 检查是否已经处理过
    if os.path.exists(output_path):
        print(f"  {file_name} already processed. Skipping...")
        return

    processed_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data['question']
                is_complex = data['planner_first_judgetoken'] == '0'

                if is_complex:
                    # 处理复杂问题
                    context = data.get('text_snippet', '')
                    prompt = f"""你是会议助手agent,根据以下会议内容回答问题，回复字数一定要在100字以内：
### 问题 ###
{question}
###

### 会议内容 ###
{context}
###

请根据以上会议内容一定只能用100字以内回答问题。不要回复无关内容,尽量简短。
"""
                else:
                    # 处理简单问题
                    prompt = f"请100字以内简要回答下面问题。越简短越好,回复字数一定要在100字以内\n 问题：{question}\n"

                answer = ask_gpt(prompt)
                
                # 构建输出数据结构
                result = {
                    "question": question,
                    "answer": answer,
                    "is_complex": is_complex,
                    "status": "completed"
                }
                if is_complex:
                    result["context"] = context
                
                # 实时写入
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line: {str(e)}")
                error_record = {
                    "question": data.get('question', ''),
                    "error": str(e),
                    "status": "failed"
                }
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(error_record, ensure_ascii=False) + '\n')

    # 记录已处理文件
    save_processed_file(file_name)

def main():
    # 获取所有jsonl文件
    jsonl_files = [os.path.join(INPUT_JSONL_DIR, f) 
                  for f in os.listdir(INPUT_JSONL_DIR) 
                  if f.endswith('.jsonl')]

    # 加载已处理文件记录
    processed_files = load_processed_files()
    
    # 过滤未处理的文件
    files_to_process = [f for f in jsonl_files 
                       if os.path.basename(f) not in processed_files]

    for jsonl_file in tqdm(files_to_process, desc="Processing JSONL files"):
        try:
            process_jsonl_file(jsonl_file)
        except Exception as e:
            print(f"Error processing file {jsonl_file}: {str(e)}")
            continue
            
    print(f"\n处理完成，结果已保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
