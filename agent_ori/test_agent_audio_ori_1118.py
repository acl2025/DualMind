import os
import threading
import numpy as np
import librosa
import whisper
from queue import Queue
from wewks.wekws.wekws.bin.stream_kws_ctc import KeyWordSpotter
from agent.agent import main as agent_main

# 修改导入方式
import agent.agent_tools

class AudioProcessor:
    def __init__(self):
        # 初始化Whisper模型
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("large")

        # 初始化关键词检测模型
        print("Loading Keyword Spotter...")

        # 检测词1
        self.kws_model_1 = KeyWordSpotter(
            ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
            config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
            token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
            lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
            threshold=0.0001,
            min_frames=3,
            max_frames=3000,
            interval_frames=30,
            score_beam=10,
            path_beam=40,
            gpu=-1,
            is_jit_model=False,
        )
        self.kws_model_1.set_keywords("交交")

        # 检测词2
        self.kws_model_2 = KeyWordSpotter(
            ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
            config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
            token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
            lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
            threshold=0.0001,
            min_frames=3,
            max_frames=3000,
            interval_frames=30,
            score_beam=10,
            path_beam=40,
            gpu=-1,
            is_jit_model=False,
        )
        self.kws_model_2.set_keywords("焦焦")

        # 检测词3
        self.kws_model_3 = KeyWordSpotter(
            ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
            config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
            token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
            lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
            threshold=0.0001,
            min_frames=3,
            max_frames=3000,
            interval_frames=30,
            score_beam=10,
            path_beam=40,
            gpu=-1,
            is_jit_model=False,
        )
        self.kws_model_3.set_keywords("教教")

        # 创建队列用于线程通信
        self.kws_queue_1 = Queue()
        self.kws_queue_2 = Queue()
        self.kws_queue_3 = Queue()
        self.stt_queue = Queue()

    def process_kws_all(self, chunk_wav):
        """并行处理所有关键词检测"""
        # 分别检测三个关键词
        kws_result_1 = self.kws_model_1.forward(chunk_wav)
        kws_result_2 = self.kws_model_2.forward(chunk_wav)
        kws_result_3 = self.kws_model_3.forward(chunk_wav)

        self.kws_queue_1.put(kws_result_1)
        self.kws_queue_2.put(kws_result_2)
        self.kws_queue_3.put(kws_result_3)

    def process_stt(self, audio_chunk):
        """语音转文字处理线程"""
        whisper_result = self.whisper_model.transcribe(
            audio_chunk,
            language='zh',
            task='transcribe',
            without_timestamps=True
        )
        self.stt_queue.put(whisper_result)

def main():
    # 设置起始时间（秒）
    start_time = 0

    audio_processor = AudioProcessor()
    #audio_path = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S02C01_agent_added_fixed/out_001-M_0.wav'
    audio_path = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S06C01_agent_added_fixed/base_add.wav'
    print(f"Processing audio file: {audio_path}")
    print(f"Starting from {start_time} seconds")

    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=16000)

    # 计算起始样本点
    start_sample = int(start_time * sr)
    if start_sample >= len(y):
        print(f"Start time {start_time}s exceeds audio duration {len(y)/sr:.2f}s")
        return

    # 从指定位置截取音频数据
    y = y[start_sample:]

    # 将音频块长度缩短到7秒，以提高响应速度和完整性
    chunk_duration = 7.0
    chunk_size = int(sr * chunk_duration)
    total_chunks = int(np.ceil(len(y) / chunk_size))

    # 初始化变量
    buffer = ""
    agent_triggered = False
    input_question = ""
    transcription = ""
    count_for_question_token = 0
    detected_keyword = ""

    # 重置所有关键词检测器
    audio_processor.kws_model_1.reset_all()
    audio_processor.kws_model_2.reset_all()
    audio_processor.kws_model_3.reset_all()

    # 在处理之前，重置 agent_tools 中的 meeting_transcript
    agent.agent_tools.meeting_transcript = ""

    print(f"Starting processing from {start_time}s, total chunks: {total_chunks}")

    # 遍历音频块
    for i in range(total_chunks):
        # 计算当前时间点
        current_time = start_time + (i * chunk_duration)

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(y))
        audio_chunk = y[start_idx:end_idx]

        # 将音频块转换为字节格式
        chunk_wav = (audio_chunk * (1 << 15)).astype("int16").tobytes()

        # 创建并启动并行处理线程
        kws_thread = threading.Thread(
            target=audio_processor.process_kws_all,
            args=(chunk_wav,)
        )
        stt_thread = threading.Thread(
            target=audio_processor.process_stt,
            args=(audio_chunk,)
        )

        kws_thread.start()
        stt_thread.start()

        # 等待两个线程完成
        kws_thread.join()
        stt_thread.join()

        # 获取处理结果
        kws_result_1 = audio_processor.kws_queue_1.get()
        kws_result_2 = audio_processor.kws_queue_2.get()
        kws_result_3 = audio_processor.kws_queue_3.get()
        whisper_result = audio_processor.stt_queue.get()

        # 处理关键词检测结果
        if any([
            'state' in kws_result_1 and kws_result_1['state'] == 1,
            'state' in kws_result_2 and kws_result_2['state'] == 1,
            'state' in kws_result_3 and kws_result_3['state'] == 1
        ]):
            # 确定是哪个关键词被检测到
            if 'state' in kws_result_1 and kws_result_1['state'] == 1:
                detected_keyword = "交交"
            elif 'state' in kws_result_2 and kws_result_2['state'] == 1:
                detected_keyword = "焦焦"
            else:
                detected_keyword = "教教"

            print(f"\n[Audio Detection] Detected keyword '{detected_keyword}' at {current_time:.2f} seconds.")
            agent_triggered = True
            # 从关键词后开始收集
            buffer = ""
            count_for_question_token = 0  # 重置计数器

        # 处理STT结果
        text = whisper_result['text'].strip()

        if text:  # 只有当有实际文本时才处理
            # 模拟逐字输出
            for index, token in enumerate(text):
                print(token, end='', flush=True)
                buffer += token
                transcription += token

                # 在这里更新 agent_tools 中的 meeting_transcript
                agent.agent_tools.append_meeting_transcript(token)

                if agent_triggered:
                    count_for_question_token += 1

                    # 当收集到70个token，或者检测到句子结束标点时，触发agent
                    if count_for_question_token >= 70 or token in ['。', '！', '？']:
                        input_question = buffer.strip()
                        if input_question:  # 确保有实际的问题内容
                            print(f"\n\n[Agent] Received question: {input_question}\n")

                            # 启动智能体处理
                            agent_thread = threading.Thread(target=agent_main, args=(input_question,))
                            agent_thread.start()
                            agent_thread.join()

                        # 重置状态
                        buffer = ""
                        agent_triggered = False
                        input_question = ""
                        count_for_question_token = 0
                        detected_keyword = ""
                        break  # 退出字符循环，处理下一个音频块

                # 文本关键字检测
                if not agent_triggered and any(keyword in buffer for keyword in ["交交", "焦焦", "教教", "娇娇", "焦家",]):
                    # 找出触发的关键词
                    for keyword in ["交交", "焦焦", "教教", "娇娇", "焦家",]:
                        if keyword in buffer:
                            detected_keyword = keyword
                            break

                    print(f"\n[Text Detection] Detected keyword '{detected_keyword}' at {current_time:.2f} seconds.")
                    agent_triggered = True
                    # 从关键词后开始收集
                    buffer = buffer.split(detected_keyword, 1)
                    if len(buffer) > 1:
                        buffer = buffer[1]
                    else:
                        buffer = ""
                    count_for_question_token = 0  # 重置计数器
        else:
            pass

    print("\n会议数据处理完成。")

    # 保存会议记录，添加起始时间信息
    output_transcript_path = f"meeting_transcript_from_{start_time}s.txt"
    with open(output_transcript_path, "w", encoding='utf-8') as f:
        f.write(f"会议记录（从 {start_time} 秒开始）：\n\n")
        # 从 agent_tools 中获取会议记录
        f.write(agent.agent_tools.meeting_transcript)
    print(f"会议记录已保存到 {output_transcript_path}")

if __name__ == "__main__":
    main()