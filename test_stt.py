from whisper_streaming.whisper_online import stt_streaming

if __name__ == "__main__":
    audio_file = "/home/leon/agent/AISHELL_dataset/insert_train_S/20200623_S_R001S07C01_agent_added/base_add.wav"
    results = stt_streaming(audio_file, min_chunk_size=1.0, lan="zh", model="large")
    # 打印或使用 results
    for item in results:
        print(item)
    # item 格式示例: (1234.56, 1000.0, 1800.0, "识别文本片段")
    # 分别表示：
    # (流式输出时刻ms, 语音开始ms, 语音结束ms, 文本内容)
