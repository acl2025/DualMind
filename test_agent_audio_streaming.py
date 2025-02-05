import os
import numpy as np
import librosa
import glob
import agent.agent_tools
import multiprocessing
import threading
import time
from threading import Thread, Event
import queue

from wewks.audio_keyword_detection import global_kws_models, init_kws_worker, kws_worker
from whisper_STT.whisper_STT import global_whisper_model, init_stt_worker, stt_worker

def main():
    from agent.agent import classifier_process_func, talker_process_func, reasoner_process_func, process_agent_trigger

    # 设置起始时间（秒）
    start_time = 0

    # 创建进程上下文，使用 'spawn' 方法
    ctx = multiprocessing.get_context('spawn')

    # 创建 Manager 和共享的 meeting_transcript
    manager = ctx.Manager()
    meeting_transcript = manager.Namespace()
    meeting_transcript.text = ""

    # 创建用于与智能体通信的队列
    classifier_input_queue = ctx.Queue()
    classifier_output_queue = ctx.Queue()

    talker_input_queue = ctx.Queue()
    talker_output_queue = ctx.Queue()

    reasoner_input_queue = ctx.Queue()
    reasoner_output_queue = ctx.Queue()

    # 启动智能体进程，并在会议数据载入前加载模型
    classifier_process = ctx.Process(target=classifier_process_func, args=(classifier_input_queue, classifier_output_queue))
    talker_process = ctx.Process(target=talker_process_func, args=(talker_input_queue, talker_output_queue))
    reasoner_process = ctx.Process(target=reasoner_process_func, args=(reasoner_input_queue, reasoner_output_queue, meeting_transcript))

    classifier_process.start()
    talker_process.start()
    reasoner_process.start()

    # 加载音频文件
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R004S02C01_agent_added_fixed/'
    
    input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S02C01_agent_added_fixed/'

    audio_path = input_folder + 'base_add.wav'

    # 新增：读取唤醒音频文件并获取长度

    wake_up_audio_dir = input_folder
    wake_up_audio_files = glob.glob(os.path.join(wake_up_audio_dir, 'out_*.wav'))

    print(f"Processing audio file: {audio_path}")
    print(f"Starting from {start_time} seconds")

    y, sr = librosa.load(audio_path, sr=16000)

    # 计算起始样本点
    start_sample = int(start_time * sr)
    if start_sample >= len(y):
        print(f"Start time {start_time}s exceeds audio duration {len(y)/sr:.2f}s")
        return

    # 从指定位置截取音频数据
    y = y[start_sample:]

    # 将音频块长度缩短到5秒，以提高响应速度和完整性
    chunk_duration = 5.0
    chunk_size = int(sr * chunk_duration)
    total_chunks = int(np.ceil(len(y) / chunk_size))

    # 初始化变量
    buffer = ""
    agent_triggered = False
    input_question = ""
    transcription = ""
    count_for_question_token = 0
    detected_keyword = ""

    # 在处理之前，重置 agent_tools 中的 meeting_transcript
    agent.agent_tools.meeting_transcript = ""

    wake_up_files_with_z = []
    for filename in wake_up_audio_files:
        basename = os.path.basename(filename)
        # 提取 Z 序号
        basename_no_ext = os.path.splitext(basename)[0]
        parts = basename_no_ext.split('_')
        if len(parts) >= 3:
            z_str = parts[-1]
            try:
                z = int(z_str)
                wake_up_files_with_z.append((z, filename))
            except ValueError:
                print(f"Cannot parse Z from filename {filename}")
        else:
            print(f"Filename {filename} does not match expected pattern")

    # 按照 Z 序号排序
    wake_up_files_with_z.sort(key=lambda x: x[0])

    # 获取每个唤醒音频的长度并存储
    wake_up_audio_lengths = []
    for z, filename in wake_up_files_with_z:
        y_wakeup, sr_wakeup = librosa.load(filename, sr=16000)
        length_in_seconds = len(y_wakeup) / sr_wakeup
        wake_up_audio_lengths.append(length_in_seconds)

    # 初始化唤醒计数器
    wake_up_counter = 0

    print(f"Starting processing from {start_time}s, total chunks: {total_chunks}")

    # 创建进程池
    with ctx.Pool(processes=1, initializer=init_kws_worker) as kws_pool, \
         ctx.Pool(processes=1, initializer=init_stt_worker) as stt_pool:

        # 等待初始化好智能体权重
        time.sleep(30)

        # 初始化 previous_text
        previous_text = ""
        max_previous_text_length = 50  # 设置 previous_text 的最大长度

        # 遍历音频块
        for i in range(total_chunks):
            # 计算当前时间点
            current_time = start_time + (i * chunk_duration)

            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(y))
            audio_chunk = y[start_idx:end_idx]

            # 将音频块转换为字节格式
            chunk_wav = (audio_chunk * (1 << 15)).astype("int16").tobytes()

            # 提交并行任务
            kws_future = kws_pool.apply_async(kws_worker, args=(chunk_wav,))
            stt_future = stt_pool.apply_async(stt_worker, args=(audio_chunk, previous_text))

            # 获取处理结果
            kws_result = kws_future.get()
            kws_result_1, kws_result_2, kws_result_3 = kws_result
            whisper_result = stt_future.get()

            # 处理关键词检测结果
            if any([
                'state' in kws_result_1 and kws_result_1['state'] == 1,
                'state' in kws_result_2 and kws_result_2['state'] == 1,
                'state' in kws_result_3 and kws_result_3['state'] == 1
            ]):
                # 确定是哪个关键词被检测到
                if 'state' in kws_result_1 and kws_result_1['state'] == 1:
                    detected_keyword = "好交交"
                elif 'state' in kws_result_2 and kws_result_2['state'] == 1:
                    detected_keyword = "好焦焦"
                else:
                    detected_keyword = "好教教"

                print(f"\n[Audio Detection] Detected keyword '{detected_keyword}' at {current_time:.2f} seconds.")

                # 调用处理函数
                process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                                      stt_pool, classifier_input_queue, classifier_output_queue,
                                      talker_input_queue, talker_output_queue,
                                      reasoner_input_queue, reasoner_output_queue,
                                      meeting_transcript, previous_text)
                # 增加唤醒计数器
                wake_up_counter += 1
                continue  # 跳过当前块的后续处理

            # 处理STT结果
            text = whisper_result['text'].strip()

            if text:  # 只有当有实际文本时才处理
                # 模拟逐字输出
                for index, token in enumerate(text):
                    print(token, end='', flush=True)
                    buffer += token
                    transcription += token

                    # 更新共享的 meeting_transcript
                    meeting_transcript.text += token

                    # 文本关键字检测
                    if not agent_triggered and any(keyword in buffer for keyword in ["交交", "焦焦", "好教教", ",教教", "娇娇", "焦家"]):
                        # 找出触发的关键词
                        for keyword in ["交交", "焦焦", "好教教", ",教教", "娇娇", "焦家"]:
                            if keyword in buffer:
                                detected_keyword = keyword
                                break

                        print(f"\n[Text Detection] Detected keyword '{detected_keyword}' at {current_time:.2f} seconds.")

                        # 调用处理函数
                        process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                                              stt_pool, classifier_input_queue, classifier_output_queue,
                                              talker_input_queue, talker_output_queue,
                                              reasoner_input_queue, reasoner_output_queue,
                                              meeting_transcript, previous_text)
                        # 增加唤醒计数器
                        wake_up_counter += 1
                        buffer = ""  # 重置 buffer
                        break  # 跳出字符循环，处理下一个音频块
                else:
                    pass

                # 更新 previous_text，并限制其长度
                previous_text += text + ' '
                if len(previous_text) > max_previous_text_length:
                    # 保留 previous_text 的最后 max_previous_text_length 个字符
                    previous_text = previous_text[-max_previous_text_length:]

    # 在程序结束时，停止智能体进程
    classifier_input_queue.put('STOP')
    talker_input_queue.put('STOP')
    reasoner_input_queue.put('STOP')

    classifier_process.join()
    talker_process.join()
    reasoner_process.join()

    print("\n会议数据处理完成。")

    # 保存会议记录，添加起始时间信息
    output_transcript_path = f"meeting_transcript_from_{start_time}s.txt"
    with open(output_transcript_path, "w", encoding='utf-8') as f:
        f.write(f"会议记录（从 {start_time} 秒开始）：\n\n")
        # 从共享的 meeting_transcript 获取会议记录
        f.write(meeting_transcript.text)
    print(f"会议记录已保存到 {output_transcript_path}")

if __name__ == "__main__":
    # 在主程序开始时设置启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()