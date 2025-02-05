from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"

# 修改模型加载参数
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    # 移除 device_map 和 torch_dtype 参数
)

# 将模型移动到 GPU
if torch.cuda.is_available():
    model = model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 确保输入数据在正确的设备上
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)