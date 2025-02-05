import os
import re

def extract_text_from_textgrid(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配 text = "..." 的模式
    pattern = r'text = "(.*?)"'
    matches = re.findall(pattern, content)

    # 过滤掉空字符串
    non_empty_texts = [text for text in matches if text.strip()]

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in non_empty_texts:
            f.write(text + '\n')

def process_all_textgrids(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.TextGrid'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)

            extract_text_from_textgrid(input_path, output_path)
            print(f"Processed {filename} -> {output_filename}")

# 使用示例
input_folder = '/home/leon/agent/AISHELL_dataset/test/TextGrid'
output_folder = '/home/leon/agent/AISHELL_dataset/test/text_raw'

process_all_textgrids(input_folder, output_folder)
print("All TextGrid files have been processed.")