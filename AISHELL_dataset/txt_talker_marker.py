import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载Qwen2.5模型和分词器
model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
#model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义输入和输出文件夹路径
input_dir = "/home/leon/agent/AISHELL_dataset/test/text_raw"
output_dir = "/home/leon/agent/AISHELL_dataset/test/text_marked"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 设置模型的最大输入长度
max_input_length = 500  # 根据需要调整

# 定义函数，将内容按最大长度分割成块
def split_into_chunks(lines, max_input_length):
    chunks = []
    current_chunk = []
    current_length = 0
    for line in lines:
        line_tokens = tokenizer(line, return_tensors='pt', add_special_tokens=False).input_ids.shape[1]
        if current_length + line_tokens <= max_input_length:
            current_chunk.append(line)
            current_length += line_tokens
        else:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_tokens
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_dir, filename)
        file_prefix = os.path.splitext(filename)[0]
        output_filename = file_prefix + "_marked.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将内容按行分割
        lines = content.splitlines()
        # 将内容分割成符合最大输入长度的块
        chunks = split_into_chunks(lines, max_input_length)
        
        annotated_content = ''
        for chunk in chunks:
            # 准备提示语
            prompt = """ 请将以下会议记录标注每一次数据是哪个人发言，如果不确定就标注为未知或杂音：\n
            ### 示例输入：###
            这块儿<#>&呃&别的看看有没有这个呃负责人对这个车辆，这块儿有的业主有没有提了一些呃想法
            这个想法来定一个方案，对吧
            嗯
            就这块儿的话，咱们车位，咱们不是说比较难解决嘛，有没有更好更好&嗯&的建议
            咱这块儿把这个，业主这边儿这个提上来你看，车位不够，呃你看有的也说这个家里有两三辆车，对吧，呃<sil>还有的说回来呃自己的车位让别人占了
            咱小区啊没有不分，车位是自己的这个，除非人家买下来，买下来的话这个车位属于人家的嗯，他自己会管理自己的车位，比如在他自己地方加把锁对吧，这是他自己的车位，专车
            ###
            
            ### 示例输出：###
            一号发言者：这块儿<#>&呃&别的看看有没有这个呃负责人对这个车辆，这块儿有的业主有没一号发言者：有提了一些呃想法 
            一号发言者：这个想法来定一个方案，对吧
            二号发言者：嗯 
            二号发言者：就这块儿的话，咱们车位，咱们不是说比较难解决嘛，有没有更好更好&嗯&的建议
            二号发言者：咱这块儿把这个，业主这边儿这个提上来你看，车位不够，呃你看有的也说这个家二号发言者：里有两三辆车，对吧，呃<sil>还有的说回来呃自己的车位让别人占了
            一号发言者：咱小区啊没有不分，车位是自己的这个，除非人家买下来，买下来的话这个车位属于人家的嗯，他自己会管理自己的车位，比如在他自己地方加把锁对吧，这是他自己的车位，专车
            ###
            
            ### 待处理的会议记录 ###
            
            """  + chunk +"""\n ### """
            print("-------prompt: ", prompt)
            
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # 生成标注后的内容
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print("-------response: ", response)
            
            annotated_content += response + '\n'
        
        # 将标注后的内容写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(annotated_content)
