import os
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 如果不需要使用 tiktoken 计算 tokens，可直接删除相关代码。
# 若保留此功能（如在后续有需要统计 token），可根据实际情况适配。
# 这里先注释掉，与 Llama-2-7b-chat-hf 不冲突，但也不会使用。
# import tiktoken

# --- 下面的行与 OpenAI API 调用相关，已全部注释掉 ---
# import openai
# openai.api_key = 'YOUR_API_KEY'

# INPUT & OUTPUT 目录常量
INPUT_JSONL_DIR = '/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103'
OUTPUT_DIR = '/home/leon/agent/baseline/result_Llama-2-7b-chat-hf_baseline'
PROCESSED_RECORD = os.path.join(OUTPUT_DIR, '.processed_files')

# 如果仍想用 tiktoken 做分词统计，可以保留并指定适配的 Llama tokenizer。
# 否则可以删除以下代码及调用。
# encoding = tiktoken.encoding_for_model("gpt-4-0125-preview")

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

def ask_llama(prompt):
    """
    使用 Llama-2-7b-chat-hf 来回答问题。
    每次调用时都重新加载模型，并放到 GPU 上（如果可用）。
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",        # 或者手动 model.to("cuda")
        torch_dtype=torch.float16
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    output = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # ---- 额外处理：移除可能重复的 prompt（可选）----
    # 如果发现答案里包含 prompt，就去掉 prompt 前的部分
    # （也可以根据自定义分隔符做更多精细化处理）
    if prompt in output:
        # 方法1：直接整段替换（可能会把 prompt 中的文字全部删掉）
        output = output.replace(prompt, "").strip()
        # 方法2：若只想保留 prompt 之后的内容，可用 split：
        # parts = output.split(prompt, 1)
        # if len(parts) > 1:
        #     output = parts[1].strip()

    return output

def process_jsonl_file(jsonl_path):
    """处理单个jsonl文件"""
    file_name = os.path.basename(jsonl_path)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    if os.path.exists(output_path):
        print(f"  {file_name} already processed. Skipping...")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            is_complex = data.get('planner_first_judgetoken', '') == '0'
            
            try:
                # 根据是否是复杂问题组装 prompt
                if is_complex:
                    context = data.get('text_snippet', '')
                    prompt = f"""你是会议助手agent,根据以下会议内容用中文回答问题，回复字数一定要在100字以内：
### 问题 ###
{question}
###

### 会议内容 ###
{context}
###

请根据以上会议内容一定只能用100字以内回答问题。不要回复无关内容,尽量简短。
"""
                else:
                    prompt = f"请100字以内用中文简要回答下面问题。越简短越好,回复字数一定要在100字以内\n 问题：{question}\n"

                print(prompt)
                # 调用 Llama
                answer_raw = ask_llama(prompt)
                
                print(answer_raw)

                # 组装输出
                result = {
                    "question": question,
                    "prompt": prompt,        # 新增：输出到 jsonl
                    "answer": answer_raw,    # 仅保留真正回答
                    "is_complex": is_complex,
                    "status": "completed"
                }

                if is_complex:
                    result["context"] = context

                # 写入输出文件
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line: {str(e)}")
                error_record = {
                    "question": data.get('question', ''),
                    "prompt": prompt,  # 如果出错，也记录一下 prompt 以便排查
                    "error": str(e),
                    "status": "failed"
                }
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(error_record, ensure_ascii=False) + '\n')

    save_processed_file(file_name)

def main():
    jsonl_files = [
        os.path.join(INPUT_JSONL_DIR, f) 
        for f in os.listdir(INPUT_JSONL_DIR) 
        if f.endswith('.jsonl')
    ]
    processed_files = load_processed_files()
    files_to_process = [
        f for f in jsonl_files 
        if os.path.basename(f) not in processed_files
    ]

    for jsonl_file in tqdm(files_to_process, desc="Processing JSONL files"):
        try:
            process_jsonl_file(jsonl_file)
        except Exception as e:
            print(f"Error processing file {jsonl_file}: {str(e)}")
            continue

    print(f"\n处理完成，结果已保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()