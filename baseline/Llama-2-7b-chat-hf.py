# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",        # 或者手动 model.to("cuda")
        torch_dtype=torch.float16
    )

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

generated_ids = model.generate(**inputs, max_new_tokens=100)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)