from agent.classifier_reasoner_talker import classifier_process_func, talker_process_func, reasoner_process_func
import time
import queue
# 定义读取智能体输出的函数
def read_agent_output(output_queue, output_buffer, stop_event):
    import queue
    while not stop_event.is_set():
        try:
            token = output_queue.get(timeout=0.1)
            if token == '__end__':
                output_buffer.append(token)
                break
            output_buffer.append(token)
        except queue.Empty:
            continue
        
# def process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
#                           stt_pool, planner_input_queue, planner_output_queue,
#                           talker_input_queue, talker_output_queue,
#                           meeting_transcript, previous_text, hard_question_flag, audio_detection_flag, y_out_question_segment):


def process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                          stt_pool, planner_input_queue, planner_output_queue,
                          talker_input_queue, talker_output_queue,
                          meeting_transcript, previous_text, input_question_text_realtime_latency, hard_question_flag, audio_detection_flag,):
    
# def process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
#                           stt_pool, classifier_input_queue, classifier_output_queue,
#                           talker_input_queue, talker_output_queue,
#                           reasoner_input_queue, reasoner_output_queue,
#                           meeting_transcript):
    
    import numpy as np
    #from whisper_STT.whisper_STT import stt_worker
    import time
    from threading import Thread
    import queue
    
    hard_question_time = 50

    if wake_up_counter < len(wake_up_audio_lengths):
        wake_up_length = wake_up_audio_lengths[wake_up_counter]
    else:
        wake_up_length = hard_question_time  # 默认持续时间
        print("没有更多的唤醒音频长度，使用默认持续时间。")
        
    if hard_question_flag and wake_up_length <= 25.0:
        wake_up_length = hard_question_time  # 默认持续时间
        print("是hard question，使用默认复杂问题持续时间。")
        
    # if not hard_question_flag and wake_up_length > 15.0:
    #     wake_up_length = 15  # 默认持续时间
    #     print("不是hard question，使用默认简单问题持续时间。")

    # 计算开始和结束时间
    start_time = current_time 
    end_time = current_time + wake_up_length 

    # 确保结束时间不超过总时长
    total_duration = len(y) / sr
    if end_time > total_duration:
        end_time = total_duration

    # 计算采样索引
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # 提取音频片段
    question_audio_chunk = y[start_sample:end_sample]

    # 确保音频数据为 float32 格式
    question_audio_chunk = question_audio_chunk.astype(np.float32)

    # 检查音频数据是否为空
    if len(question_audio_chunk) == 0:
        print("警告：提取的音频片段为空。")
        return

    # 打印音频数据信息
    print(f"问题音频片段长度: {len(question_audio_chunk)}, dtype: {question_audio_chunk.dtype}, min: {question_audio_chunk.min()}, max: {question_audio_chunk.max()}")
    print(f"问题音频时间长度: {len(question_audio_chunk) / sr}")

    # 将音频片段发送到 Whisper STT（用于转写）
    #stt_future = stt_pool.apply_async(stt_worker, args=(question_audio_chunk,))
    # stt_future = stt_pool.apply_async(stt_worker, args=(question_audio_chunk,previous_text,))
    # whisper_result = stt_future.get()

    # 获取转写结果
    input_question = input_question_text_realtime_latency

    if input_question:
        
        print(f"\n\n[Agent] 接收到问题: {input_question}\n, {time.time()}")
        
        #if ("基于之前" in input_question or "至於之前" in input_question or "基於之前" in input_question) and wake_up_length <= 15.0 and audio_detection_flag:
        # if ("基于之前" in input_question or "至於之前" in input_question or "基於之前" in input_question) and wake_up_length <= 25.0 :
        #     print("是hard question，input_question 中包含 '基于之前',问题过于短,使用默认复杂问题持续时间重新STT。")
            
        #     hard_question_flag = True
            
        #     wake_up_length = hard_question_time # 默认持续时间
        #     print(f"是hard question，使用默认复杂问题持续时间,音频长度为{wake_up_length}秒。")
        #     start_time = current_time
        #     end_time = current_time + wake_up_length
            
        #     # 确保结束时间不超过总时长
        #     total_duration = len(y) / sr
        #     if end_time > total_duration:
        #         end_time = total_duration
                
        #     # 计算采样索引
        #     start_sample = int(start_time * sr)
        #     end_sample = int(end_time * sr)
            
        #     # 提取音频片段
        #     question_audio_chunk = y[start_sample:end_sample]

        #     # 确保音频数据为 float32 格式
        #     question_audio_chunk = question_audio_chunk.astype(np.float32)
            
        #     stt_future = stt_pool.apply_async(stt_worker, args=(question_audio_chunk,previous_text,))
        #     whisper_result = stt_future.get()

        #     # 获取转写结果
        #     input_question = whisper_result['text'].strip()
            
        #----------------    
        #以下部分代码仅在test_agent_audio_experiement_audio_segment_only.py中跑实验使用,如果使用test_agent_audio.py，则不需要这部分代码
        # import time
        # time_STT_start = time.time()
        # print(f"问题音频STT前的时刻: {time_STT_start}")
        # question_audio_chunk = y_out_question_segment
        # stt_future_out_question_segment = stt_pool.apply_async(stt_worker, args=(y_out_question_segment,previous_text,))
        # whisper_result_out_question_segment = stt_future_out_question_segment.get()
        # input_question = whisper_result_out_question_segment['text'].strip()# 获取转写结果
        
        # time_STT_end = time.time()
        # print(f"问题音频STT后的时刻: {time_STT_end}")
        # print(f"问题音频STT时间: {time_STT_end - time_STT_start}")
        
        #----------------   

        print(f"\n\n[Agent] 最终接收到问题: {input_question}\n, {time.time()}")

        # 更新共享的会议记录
        meeting_transcript.text += input_question
        
        #响应实时性测试
        print(f"问题音频送入Planner的时刻: {time.time()}")

        # 将问题发送给 planner
        planner_input_queue.put(input_question)

        # 创建线程读取 planner 的输出
        def read_planner_output(output_queue, output_list):
            while True:
                token = output_queue.get()
                if token == '__end__':
                    break
                output_list.append(token)

        planner_output_list = []
        planner_thread = Thread(target=read_planner_output, args=(planner_output_queue, planner_output_list))
        planner_thread.start()
        planner_thread.join()

        # 解析 planner 的输出
        if planner_output_list:
            print(f"planner_output_list: {planner_output_list}")
            first_token = planner_output_list[0].strip()
            
            #if "1" in planner_output_list:
            if "1" in first_token:
                selected_agent = 'talker'
            #elif "0" in planner_output_list:
            elif "0" in first_token:
                selected_agent = 'planner'
            else:
                print("未能正确解析到 0/1，原始判定输出：", ''.join(planner_output_list))
                return

            print(f"\n选择的智能体：{selected_agent}")

            if selected_agent == 'talker':
                # 准备 talker 的输入
                talker_input = {
                    'audio': question_audio_chunk,
                    'text': '只用100字以内回答语音中的问题。',
                    'sr': sr  # 传递采样率
                }
                # 发送输入给 talker
                print("将问题音频输入给 talker")
                talker_input_queue.put(talker_input)

                # 创建线程读取 talker 的输出
                def read_talker_output(output_queue, output_list):
                    while True:
                        token = output_queue.get()
                        if token == '__end__':
                            break
                        output_list.append(token)

                talker_output_list = []
                talker_thread = Thread(target=read_talker_output, args=(talker_output_queue, talker_output_list))
                talker_thread.start()
                talker_thread.join()

                # 在主进程中打印 talker 的输出
                talker_output_str = ''.join(talker_output_list)
                print("\ntalker输出：")
                print(talker_output_str)
                print("talker 输出结束")

            else:
                # 继续处理 planner 的输出
                planner_remaining_output = ''.join(planner_output_list[-1]).strip()
                print(f"{planner_remaining_output}")
                print("===planner 输出结束===")


        else:
            print("planner 没有产生输出")


if __name__ == "__main__":
    pass  # 主程序中不执行任何操作