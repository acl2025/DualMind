import os
import subprocess
import re
import json
import time

def main():
    # 输入和输出目录
    input_root = '/home/leon/agent/AISHELL_dataset/insert_train_S'
    output_root = '/home/leon/agent/experiment_result/result_train_S_audio_segment_only_0117_real_time_latency'

    # 日志文件，用于断点续传
    log_file = os.path.join(output_root, 'processed_files.log')
    processed_files = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_files = set(f.read().splitlines())

    # 获取所有会议文件夹
    meeting_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    for meeting_name in meeting_dirs:
        meeting_dir = os.path.join(input_root, meeting_name)
        audio_file = os.path.join(meeting_dir, 'base_add.wav')

        # 检查音频文件是否存在
        if not os.path.exists(audio_file):
            print(f"音频文件不存在：{audio_file}")
            continue

        # 检查是否已处理
        if meeting_name in processed_files:
            print(f"已处理，跳过：{meeting_name}")
            continue

        print(f"处理会议：{meeting_name}")

        # 输出文件路径
        txt_output_path = os.path.join(output_root, f"result_{meeting_name}.txt")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

        # 运行test_agent_audio.py，重定向输出
        # cmd = [
        #     'python', 'agent/test_agent_audio.py', 
        #     '--input_folder', meeting_dir
        # ]
        
        cmd = [
            'python', 'agent/test_agent_audio_experiement_audio_segment_only.py', 
            '--input_folder', meeting_dir
        ]


        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # 实时解析输出并写入文件
        with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
            buffer = ''
            for line in iter(process.stdout.readline, ''):
                print(line.strip())  # 打印实时输出
                buffer += line

                # 实时写入TXT文件
                txt_file.write(line)
                txt_file.flush()

                # 智能体输出
                if 'talker输出：' in line:
                    # 收集talker输出
                    talker_output = collect_agent_output(process.stdout, txt_file)
                elif 'reasoner输出：' in line:
                    # 收集reasoner输出
                    reasoner_output = collect_agent_output(process.stdout, txt_file)

                # 检测处理完成标志
                if '会议数据处理完成。' in line:
                    # 终止子进程
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

                    # 清理 GPU 显存
                    clear_gpu_memory()

                    break  # 结束当前会议的处理

        # 更新已处理的会议列表
        with open(log_file, 'a') as f:
            f.write(meeting_name + '\n')

def clear_gpu_memory():
    # 清理 GPU 显存的函数，根据具体环境可能需要调整
    try:
        import torch
        torch.cuda.empty_cache()
        # 如果使用了其他库的 GPU，如 TensorFlow，需要相应地清理
    except ImportError:
        pass
    # 如果需要进一步的清理，可以考虑重启特定的服务或者使用其他工具

def collect_agent_output(stdout, txt_file):
    output_lines = []
    for line in stdout:
        txt_file.write(line)
        txt_file.flush()
        output_lines.append(line)
        if 'talker 输出结束' in line or 'reasoner 输出结束' in line:
            break
    output = ''.join(output_lines)
    return output

if __name__ == "__main__":
    main()