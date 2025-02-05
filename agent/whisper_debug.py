import whisper
import numpy as np
import time

def transcribe_audio_streaming(file_path, model_name="large", chunk_length=5, chunk_overlap=1):
    """
    使用 Whisper 模型对 WAV 文件进行流式转录。

    Args:
        file_path (str): WAV 文件路径
        model_name (str): Whisper 模型名称 ("tiny", "base", "small", "medium", "large")

    Returns:
        str: 转录结果
    """
    try:
        # 使用 Whisper 的函数加载并重采样音频
        audio_data = whisper.load_audio(file_path)
        sample_rate = 16000  # Whisper 使用 16000 Hz

        # 归一化音频以保持一致的音量
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude

        # 加载 Whisper 模型
        print(f"正在加载 Whisper {model_name} 模型...")
        model = whisper.load_model(model_name)

        # 初始化变量
        transcription = ""
        chunk_samples = int(chunk_length * sample_rate)
        start_sample = 0
        previous_text = ""

        # 分块处理音频数据
        while start_sample < len(audio_data):
            end_sample = start_sample + chunk_samples
            if end_sample > len(audio_data):
                end_sample = len(audio_data)
            chunk = audio_data[start_sample:end_sample]

            time_start = time.time()
            # 调用 model.transcribe() 处理每个块
            result = model.transcribe(
                chunk,
                language='zh',
                task='transcribe',
                no_speech_threshold=0.1,
                logprob_threshold=-1.0,
                condition_on_previous_text=False,
                initial_prompt=previous_text
            )
            time_end = time.time()
            print(f"转录单个token: {time_end - time_start:.2f} 秒")
            # 获取当前文本
            current_text = result["text"]

            # 处理重叠部分，避免重复文本
            if previous_text and current_text.startswith(previous_text.strip()):
                current_text = current_text[len(previous_text.strip()):]

            # 添加到转录结果
            transcription += current_text
            
            print(current_text)

            # 更新 previous_text
            previous_text += current_text

            # 更新起始样本位置，考虑重叠
            start_sample = end_sample 

        return transcription
    except Exception as e:
        raise Exception(f"转录失败: {str(e)}")

def main():
    # 示例使用
    try:
        file_path = "/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R004S02C01_agent_added_fixed/out_001-M_0_0.wav"
        
        result = transcribe_audio_streaming(file_path)
        
        

        # 打印转录结果
        print("\n转录结果:")
        print(result)

    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
