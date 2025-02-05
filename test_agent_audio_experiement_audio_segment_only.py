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
import argparse

from wewks.audio_keyword_detection import global_kws_models, init_kws_worker, kws_worker
from whisper_STT.whisper_STT import global_whisper_model, init_stt_worker, stt_worker


def main():
    from agent.agent import process_agent_trigger
    from agent.classifier_reasoner_talker import planner_process_func, talker_process_func

    # 设置起始时间（秒）
    start_time = 0

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

    # 加载音频文件L_R003S04C02_agent_added_fixed
    #input_folder='/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S06C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R004S02C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R003S01C02_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R003S04C02_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S03C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/M_R003S01C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/M_R003S02C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/M_R003S04C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/M_R003S05C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R003S01C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R003S02C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R003S03C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R004S03C01_agent_added_fixed/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/S_R004S04C01_agent_added_fixed/'
    
    
    #input_folder = '/home/leon/agent/AISHELL_dataset/insert_train_L/20200706_L_R001S01C01_agent_added/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/insert_train_L/20200706_L_R001S02C01_agent_added/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/insert_train_L/20200706_L_R001S03C01_agent_added/'
    #input_folder = '/home/leon/agent/AISHELL_dataset/insert_train_L/20200706_L_R001S04C01_agent_added/'
    
    
    
    
    #audio_path = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R003S04C02_agent_added_fixed/base_add.wav'
    audio_path = input_folder+'/base_add.wav'
    
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

    # 将音频块长度缩短到3秒，以提高响应速度和完整性
    chunk_duration = 31.0
    
    if input_folder == '/home/leon/agent/AISHELL_dataset/insert_train_L/20200709_L_R002S05C01_agent_added':
        chunk_duration = 3.0
    
    chunk_size = int(sr * chunk_duration)
    total_chunks = int(np.ceil(len(y) / chunk_size))

    # 确保音频块大小至少为267个样本
    min_chunk_size = 267
    if chunk_size < min_chunk_size:
        chunk_size = min_chunk_size
        chunk_duration = chunk_size / sr
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
        time.sleep(25)
        
        # 初始化 previous_text
        previous_text = ""
        max_previous_text_length = 50  # 设置 previous_text 的最大长度

        # 添加跳过计数器
        skip_chunks = 0
        
        
        
        # 添加一个新变量来跟踪上次触发的时间
        last_trigger_time = -float('inf')
        # 设置最小触发间隔（秒）
        MIN_TRIGGER_INTERVAL = 10.0  

        # 遍历音频块
        for i in range(total_chunks):
            # 添加一个flag，用于标记是否是hard question
            hard_question_flag = False
            
            #添加一个flag，用于标记是否是audio detection
            audio_detection_flag = False
            
            # 如果需要跳过当前块
            if skip_chunks > 0:
                skip_chunks -= 1
                continue

            # 计算当前时间点
            current_time = start_time + (i * chunk_duration)

            # 检查是否已经过了足够的时间间隔
            if current_time - last_trigger_time < MIN_TRIGGER_INTERVAL:
                continue

            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(y))
            audio_chunk = y[start_idx:end_idx]

            # 确保音频块长度符合要求
            if len(audio_chunk) < min_chunk_size:
                # 如果最后一个块太小，用0填充到最小长度
                audio_chunk = np.pad(audio_chunk, (0, min_chunk_size - len(audio_chunk)))

            # 将音频块转换为字节格式
            chunk_wav = (audio_chunk * (1 << 15)).astype("int16").tobytes()

            # 提交并行任务
            kws_future = kws_pool.apply_async(kws_worker, args=(chunk_wav,))
            stt_future = stt_pool.apply_async(stt_worker, args=(audio_chunk, previous_text ))

            # 获取处理结果
            kws_result = kws_future.get()
            kws_result_1, kws_result_2, kws_result_3 = kws_result
            whisper_result = stt_future.get()

            # 处理关键词检测结果 audio detection
            if any([
                kws_result_1.get('state') == 1,
                kws_result_2.get('state') == 1,
                kws_result_3.get('state') == 1
            ]):
                last_trigger_time = current_time
                # 确定是哪个关键词被检测到，并调整 current_time
                if kws_result_1.get('state') == 1:
                    detected_keyword = kws_result_1['keyword']
                    detection_time = current_time + kws_result_1['start'] + 0.5
                elif kws_result_2.get('state') == 1:
                    detected_keyword = kws_result_2['keyword']
                    detection_time = current_time + kws_result_2['start'] + 0.5
                else:
                    detected_keyword = kws_result_3['keyword']
                    detection_time = current_time + kws_result_3['start'] + 0.5

                print(f"\n[Audio Detection] 在 {detection_time:.2f} 秒检测到关键词 '{detected_keyword}'。")
                print(f"detection_time: {detection_time:.2f}, current_time: {current_time:.2f}")
                
                    
                audio_detection_flag = True
                
                # # 调用处理函数
                # process_agent_trigger(detection_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                #                       stt_pool, planner_input_queue, planner_output_queue,talker_input_queue, talker_output_queue,
                #                       meeting_transcript, previous_text, hard_question_flag, audio_detection_flag)
                # 增加唤醒计数器
                wake_up_counter += 1
                # 设置跳过接下来的两个块
                skip_chunks = 1
                continue  # 跳过当前块的后续处理

            # 处理STT结果
            text = whisper_result['text'].strip()

            if text:  # 只有当有实际文本时才处理,text detection
                char_duration = chunk_duration / len(text) if len(text) > 0 else 0
                # 模拟逐字输出
                for index, token in enumerate(text):
                    print(token, end='', flush=True)
                    buffer += token
                    transcription += token

                    # 更新共享的 meeting_transcript
                    meeting_transcript.text += token

                    # 文本关键字检测时也要检查时间间隔
                    if not agent_triggered and current_time - last_trigger_time >= MIN_TRIGGER_INTERVAL:
                        for keyword in ["好交交", "焦焦", "好教教", ",教教", "娇娇", "焦家", "你好交", "佼佼", "好交", "好焦","你好焦","你好教","基于之前","至於之前","基於之前","際於之前"]:
                            keyword_pos = text.rfind(keyword)
                            if keyword_pos != -1:
                                last_trigger_time = current_time  # 更新最后触发时间
                                detected_keyword = keyword
                                # 计算关键词出现的相对时间
                                detection_time = current_time + ((keyword_pos + 1 )   * char_duration)

                                
                                if keyword == "基于之前" or keyword == "至於之前" or keyword == "基於之前" or keyword == "際於之前":
                                    detection_time = current_time + ((keyword_pos + 1 )   * char_duration) - 2
                                    hard_question_flag = True
                                    
                                if "基于之前" in text or "至於之前" in text or "基於之前" in text or "際於之前" in text:
                                    hard_question_flag = True
                                #detection_time = current_time 
                                print(f"\n[Text Detection] 在 {detection_time:.2f} 秒检测到关键词 '{detected_keyword}',current_time: {current_time:.2f}, 问题文本: {text}")
                                
                                # # 调用处理函数
                                # process_agent_trigger(detection_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                                #                       stt_pool, planner_input_queue, planner_output_queue,talker_input_queue, talker_output_queue,
                                #                       meeting_transcript, previous_text, hard_question_flag, audio_detection_flag)
                                # 增加唤醒计数器
                                wake_up_counter += 1
                                buffer = ""  # 重置 buffer
                                # 设置跳过接下来的两个块
                                skip_chunks = 1
                                break  # 跳出关键词检测循环
                
                # 更新 previous_text，并限制其长度
                previous_text += text + ' '
                if len(previous_text) > max_previous_text_length:
                    # 保留 previous_text 的最后 max_previous_text_length 个字符
                    previous_text = previous_text[-max_previous_text_length:]
                    
        #以下部分代码仅在test_agent_audio_experiement_audio_segment_only.py中跑实验使用   
        print("会议内容转录完毕。处理会议问题")         
        # 处理input_folder下的out开头的wav文件
        out_wav_files = glob.glob(os.path.join(input_folder, 'out_*.wav'))
        
        # 按照文件名最后的数字顺序排序
        out_wav_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        
        # 处理每个out文件
        for out_file in out_wav_files:
            print(f"\n处理文件: {out_file}")
            # 读取音频文件
            y_out_question_segment, sr_out = librosa.load(out_file, sr=16000)
            
            # 调用处理函数
            process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                                stt_pool, planner_input_queue, planner_output_queue,talker_input_queue, talker_output_queue,
                                meeting_transcript, previous_text, hard_question_flag, audio_detection_flag, y_out_question_segment)
            
            
            
        


    # 在程序结束时，停止智能体进程
    planner_input_queue.put('STOP')
    talker_input_queue.put('STOP')

    planner_process.join()
    talker_process.join()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help='输入音频文件夹路径')
    args = parser.parse_args()

    input_folder = args.input_folder
    audio_path = os.path.join(input_folder, 'base_add.wav')
    main()
