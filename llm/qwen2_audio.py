import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor 
#The code of Qwen2-Audio has been in the latest Hugging face transformers and we advise you to build from source with command pip install git+https://github.com/huggingface/transformers, or you might encounter the following error:

import librosa
import numpy as np

# 初始化模型和处理器
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct") 
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto",torch_dtype="auto", ) 

try:
    # 准备音频数据
    audio_path = "/home/leon/agent/agent/20241203_audio.wav"
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    print(f"Loaded audio file. Shape: {speech_array.shape}, Sample rate: {sampling_rate}")

    # 构建对话历史，添加 audio_url
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "100字以内回答该问题。"},
                {"type": "audio", "audio_url": "local_audio_1"}  # 添加 audio_url
            ]
        }
    ]

    # 准备文本输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Generated text:", text)  # 输出生成的文本，检查是否包含音频标记

    # 构建 audios 列表，匹配 audio_url
    audios = []
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if ele['audio_url'] == 'local_audio_1':
                        audios.append(speech_array)
                    else:
                        # 如果有其他音频，可以在这里添加处理逻辑
                        pass

    # 创建模型输入
    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
    )

    # 将所有张量移到 CUDA
    device = model.device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("\nInput shapes:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")

    # 打印详细的输入信息用于调试
    print("\nDetailed input information:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}:")
            print(f"  Shape: {v.shape}")
            print(f"  Device: {v.device}")
            print(f"  Type: {v.dtype}")
            if k == "input_ids":
                print("  First few tokens:", processor.tokenizer.convert_ids_to_tokens(v[0][:10].tolist()))

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1
        )

    # 解码输出
    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\nResponse:", response)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
