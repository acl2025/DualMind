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


def process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
                          stt_pool, classifier_input_queue, classifier_output_queue,
                          talker_input_queue, talker_output_queue,
                          reasoner_input_queue, reasoner_output_queue,
                          meeting_transcript,previous_text, hard_question_flag, audio_detection_flag):
    
# def process_agent_trigger(current_time, y, sr, wake_up_audio_lengths, wake_up_counter,
#                           stt_pool, classifier_input_queue, classifier_output_queue,
#                           talker_input_queue, talker_output_queue,
#                           reasoner_input_queue, reasoner_output_queue,
#                           meeting_transcript):
    
    import numpy as np
    from whisper_STT.whisper_STT import stt_worker
    import time
    from threading import Thread
    import queue

    if wake_up_counter < len(wake_up_audio_lengths):
        wake_up_length = wake_up_audio_lengths[wake_up_counter]
    else:
        wake_up_length = 45.0  # 默认持续时间
        print("没有更多的唤醒音频长度，使用默认持续时间。")
        
    if hard_question_flag and wake_up_length < 15.0:
        wake_up_length = 45.0  # 默认持续时间
        print("是hard question，使用默认复杂问题持续时间。")
        
    if not hard_question_flag and wake_up_length > 15.0:
        wake_up_length = 15  # 默认持续时间
        print("不是hard question，使用默认简单问题持续时间。")

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
    stt_future = stt_pool.apply_async(stt_worker, args=(question_audio_chunk,previous_text,))
    whisper_result = stt_future.get()

    # 获取转写结果
    input_question = whisper_result['text'].strip()

    if input_question:
        
        print(f"\n\n[Agent] 接收到问题: {input_question}\n, {time.time()}")
        
        if ("基于之前" in input_question or "至於之前" in input_question or "基於之前" in input_question) and wake_up_length <= 15.0 and audio_detection_flag:
            print("是hard question，input_question 中包含 '基于之前',问题过于短,使用默认复杂问题持续时间重新STT。")
            
            hard_question_flag = True
            
            wake_up_length = 45.0  # 默认持续时间
            print(f"是hard question，使用默认复杂问题持续时间,音频长度为{wake_up_length}秒。")
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
            
            stt_future = stt_pool.apply_async(stt_worker, args=(question_audio_chunk,previous_text,))
            whisper_result = stt_future.get()

            # 获取转写结果
            input_question = whisper_result['text'].strip()


        print(f"\n\n[Agent] 最终接收到问题: {input_question}\n, {time.time()}")

        # 更新共享的会议记录
        meeting_transcript.text += input_question

        # 将问题发送给 classifier
        classifier_input_queue.put(input_question)

        # 获取 classifier 的输出
        classifier_output = classifier_output_queue.get()

        print(f"\nclassifier的输出：{classifier_output}")

        # 根据输出，选择智能体
        if "1" in classifier_output:
            selected_agent = 'talker'
        else:
            selected_agent = 'reasoner'

        print(f"\n选择的智能体：{selected_agent}")

        if selected_agent == 'talker':
            # 准备 talker 的输入
            talker_input = {
                'audio': question_audio_chunk,
                'text': '只用100字以内回答语音中的问题。',
                'sr': sr  # 传递采样率
            }
            # 发送输入到 talker
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
            # 将问题发送给 reasoner
            reasoner_input_queue.put(input_question)

            # 创建线程读取 reasoner 的输出
            def read_reasoner_output(output_queue, output_list):
                while True:
                    token = output_queue.get()
                    if token == '__end__':
                        break
                    output_list.append(token)

            reasoner_output_list = []
            reasoner_thread = Thread(target=read_reasoner_output, args=(reasoner_output_queue, reasoner_output_list))
            reasoner_thread.start()
            reasoner_thread.join()

            # 在主进程中打印 reasoner 的输出
            reasoner_output_str = ''.join(reasoner_output_list)
            print("\nreasoner输出：")
            print(reasoner_output_str)
            print("reasoner 输出结束")


if __name__ == "__main__":
    pass  # 主程序中不执行任何操作