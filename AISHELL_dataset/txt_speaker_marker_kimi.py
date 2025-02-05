import os
from openai import OpenAI

# 使用Kimi Chat的API
client = OpenAI(
    api_key="sk-4Stc5dY49tGPqqcvMktNLDjeRpAlougEyKwX42Dlns6x4XiX",  # 请在此处替换为你从Kimi开放平台申请的API Key
    base_url="https://api.moonshot.cn/v1",
)

# 定义输入和输出文件夹路径
input_dir = "/home/leon/agent/AISHELL_dataset/test/text_raw"
output_dir = "/home/leon/agent/AISHELL_dataset/test/text_marked"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 设置模型的最大输入长度（根据API支持的最大长度调整）
max_input_length = 1000  # moonshot-v1-8k模型支持8000个token

# 定义函数，将内容按最大长度分割成块
def split_into_chunks(text, max_length):
    chunks = []
    current_chunk = ""
    for line in text.splitlines():
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
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
        
        # 将内容分割成符合最大输入长度的块
        chunks = split_into_chunks(content, max_input_length)
        
        annotated_content = ''
        for chunk in chunks:
            # 准备提示语
            prompt = f"""以下会议记录缺少发言者是谁的标注，请将以下会议记录标注每一次数据是哪个人发言，如果不确定就标注为未知或杂音：

### 示例输入：###
这块儿<#>&呃&别的看看有没有这个呃负责人对这个车辆，这块儿有的业主有没有提了一些呃想法
这个想法来定一个方案，对吧
嗯
就这块儿的话，咱们车位，咱们不是说比较难解决嘛，有没有更好更好&嗯&的建议
咱这块儿把这个，业主这边儿这个提上来你看，车位不够，呃你看有的也说这个家里有两三辆车，对吧，呃<sil>还有的说回来呃自己的车位让别人占了
咱小区啊没有不分，车位是自己的这个，除非人家买下来，买下来的话这个车位属于人家的嗯，他自己会管理自己的车位，比如在他自己地方加把锁对吧，这是他自己的车位，专车
###

### 示例输出：###
一号发言者：这块儿<#>&呃&别的看看有没有这个呃负责人对这个车辆，这块儿有的业主有没有提了一些呃想法 
一号发言者：这个想法来定一个方案，对吧
二号发言者：嗯 
二号发言者：就这块儿的话，咱们车位，咱们不是说比较难解决嘛，有没有更好更好&嗯&的建议
二号发言者：咱这块儿把这个，业主这边儿这个提上来你看，车位不够，呃你看有的也说这个家里有两三辆车，对吧，呃<sil>还有的说回来呃自己的车位让别人占了
一号发言者：咱小区啊没有不分，车位是自己的这个，除非人家买下来，买下来的话这个车位属于人家的嗯，他自己会管理自己的车位，比如在他自己地方加把锁对吧，这是他自己的车位，专车
###

### 待处理的会议记录 ###

{chunk}
###
"""

            print("-------prompt: ", prompt)
            # 调用Kimi Chat API
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {
                        "role": "system",
                        "content": "你是 Kimi人工智能助手，你更擅长判断会议记录是谁发言并标注每一次数据是哪个人发言",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            
            # 获取回复内容
            response = completion.choices[0].message.content.strip()
            print("-------response: ", response)
            
            annotated_content += response + '\n'
            
            
        
        # 将标注后的内容写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(annotated_content)
