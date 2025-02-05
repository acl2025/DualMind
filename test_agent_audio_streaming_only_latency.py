import os, signal
import numpy as np
import librosa
import glob
import agent.agent_tools
import multiprocessing
import threading
import time
from threading import Thread, Event
import queue
import argparse

from wewks.audio_keyword_detection import global_kws_models, init_kws_worker, kws_worker

# 导入 stt_streaming 函数
from whisper_streaming.whisper_online import stt_streaming

# 将 run_stt_streaming_process 函数移到主程序外部
def run_stt_streaming_process(audio_path, output_queue, start_time):
    stt_streaming(
        audio_path=audio_path,
        output_queue=output_queue,
        start_time=start_time,
        
    )

def main():
    from agent.agent_realtime_latency import process_agent_trigger
    from agent.classifier_reasoner_talker import planner_process_func, talker_process_func

    # 设置起始时间（秒）第一个简单问题590s附近，第二个复杂问题1082s附近 ,还要改截取问题长度，在本代码检索for i in range(
    start_time = 650

    # 创建进程上下文，使用 'spawn' 方法
    ctx = multiprocessing.get_context('spawn')

    # 创建 Manager 和共享的 meeting_transcript
    manager = ctx.Manager()
    meeting_transcript = manager.Namespace()
    meeting_transcript.text = ""

    # 创建用于与智能体通信的队列
    planner_input_queue = ctx.Queue()
    planner_output_queue = ctx.Queue()

    talker_input_queue = ctx.Queue()
    talker_output_queue = ctx.Queue()

    # 启动智能体进程，并在会议数据载入前加载模型
    planner_process = ctx.Process(target=planner_process_func, args=(planner_input_queue, planner_output_queue, meeting_transcript))
    talker_process = ctx.Process(target=talker_process_func, args=(talker_input_queue, talker_output_queue))

    planner_process.start()
    talker_process.start()

    # 获取输入文件夹和音频路径
    input_folder = ""
    if input_folder == "":
        input_folder = "/home/leon/agent/AISHELL_dataset/insert_train_S/20200623_S_R001S07C01_agent_added"
    
    audio_path = os.path.join(input_folder, 'base_add.wav')

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

    # 初始化变量
    buffer = ""
    agent_triggered = False
    input_question = ""
    transcription = ""
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

    print(f"Starting processing from {start_time}s")

    # 创建进程池
    with ctx.Pool(processes=1, initializer=init_kws_worker) as kws_pool:

        # 等待初始化好智能体权重
        time.sleep(10)

        # 初始化 previous_text
        previous_text = ""
        max_previous_text_length = 50  # 设置 previous_text 的最大长度

        # 修改部分：使用 multiprocessing.Queue 并启动新进程运行 stt_streaming
        stt_output_queue = ctx.Queue()

        # 启动 stt_streaming 进程
        stt_process = ctx.Process(target=run_stt_streaming_process, args=(audio_path, stt_output_queue, start_time))
        stt_process.start()

        # 开始处理流程
        while True:
            # 检查是否还有 STT 输出
            try:
                stt_result = stt_output_queue.get(timeout=0.1)
                if stt_result is None:
                    # 接收到结束信号，跳出循环
                    break
                emit_time_ms, beg_ts, end_ts, text = stt_result
                #print(f"emit_time_ms: {emit_time_ms}, beg_ts: {beg_ts}, end_ts: {end_ts}, text: {text}")
                now_s = emit_time_ms / 1000.0
                text = text.strip()
                print(f"[{now_s:.2f}s] {text}")

                # 更新 meeting_transcript
                meeting_transcript.text += text + ' '

                # 检查关键词
                for keyword in ["好交交", "焦焦", "好教教", ",教教", "娇娇","嬌嬌", "焦家", "你好交", "佼佼", "好交", "好焦","你好焦","你好教","基于之前","至於之前","基於之前","際於之前"]:
                    if keyword in text:
                        detected_keyword = keyword
                        detection_time = beg_ts / 1000.0
                        print(f"\n[Text Detection] 在 {detection_time:.2f} 秒检测到关键词 '{detected_keyword}'。")
                        # 获取关键词后的文本内容
                        keyword_index = text.index(keyword) + len(keyword)
                        input_question_text_realtime_latency = text[keyword_index:]
                        
                        


                        
                        for i in range(15):
                            stt_result = stt_output_queue.get(timeout=10)
                            emit_time_ms, beg_ts, end_ts, text = stt_result
                            print(f"提取问题[{emit_time_ms / 1000:.2f}s] {text}，{time.time()}")
                            input_question_text_realtime_latency += text
                            
                        # 暂停 stt_process    
                        os.kill(stt_process.pid, signal.SIGSTOP)
                        # 调用处理函数
                        process_agent_trigger(detection_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                                              None, planner_input_queue, planner_output_queue, talker_input_queue, talker_output_queue,
                                              meeting_transcript, previous_text,input_question_text_realtime_latency, hard_question_flag=False, audio_detection_flag=False,)
                        # 增加唤醒计数器
                        wake_up_counter += 1
                        
                        # 恢复 stt_process
                        os.kill(stt_process.pid, signal.SIGCONT)
                        
                        break
                # 更新 previous_text，并限制其长度
                previous_text += text + ' '
                if len(previous_text) > max_previous_text_length:
                    # 保留 previous_text 的最后 max_previous_text_length 个字符
                    previous_text = previous_text[-max_previous_text_length:]
            except queue.Empty:
                # 超时，无新输出，继续循环
                pass

if __name__ == "__main__":
    main()