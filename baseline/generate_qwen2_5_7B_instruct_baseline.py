import os
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# INPUT & OUTPUT 目录常量
INPUT_JSONL_DIR = '/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103'
# 将输出目录改名
OUTPUT_DIR = '/home/leon/agent/baseline/result_Qwen2.5-7B-Instruct_baseline'
PROCESSED_RECORD = os.path.join(OUTPUT_DIR, '.processed_files')

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

def ask_qwen2_5_7B(prompt):
    """
    使用 Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 来回答问题（仅文本 Prompt）。
    每次调用时都重新加载模型，并放到 GPU 上（如果可用）。
    """
    # 加载tokenizer和模型
    model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 构建 messages，这里只放纯文本
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # 将对话转化为可被模型处理的文本
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 构建输入
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
    )

    # 移动到模型所在设备
    device = model.device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1
        )

    # 解码输出（跳过输入部分）
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # 如果生成内容中意外包含原始 prompt，可酌情处理
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response

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
                    prompt = (
                        f"请100字以内用中文简要回答下面问题。"
                        f"越简短越好,回复字数一定要在100字以内\n"
                        f"问题：{question}\n"
                    )

                print(prompt)
                # 调用 Qwen2.5-7B-Instruct-GPTQ-Int4
                answer_raw = ask_qwen2_5_7B(prompt)
                
                print(answer_raw)

                # 组装输出
                result = {
                    "question": question,
                    "prompt": prompt,        # 新增：输出到 jsonl 以便检查
                    "answer": answer_raw,    # 模型的回答
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
                    "prompt": prompt,
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
