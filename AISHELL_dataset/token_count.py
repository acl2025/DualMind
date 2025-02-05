from transformers import AutoTokenizer
import os

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def process_file(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return count_tokens(content, tokenizer)

def process_directory(directory_path, tokenizer):
    total_tokens = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            file_tokens = process_file(file_path, tokenizer)
            print(f"{filename}: {file_tokens} tokens")
            total_tokens += file_tokens
    return total_tokens

# 使用 Qwen 模型的 tokenizer，你也可以根据需要更换为其他模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")

# 指定要处理的目录
directory_path = "/home/leon/agent/AISHELL_dataset/test/text_raw"

total_tokens = process_directory(directory_path, tokenizer)
print(f"Total tokens in all files: {total_tokens}")