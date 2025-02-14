import wave
import os

def concatenate_wav_files(input_file1, input_file2, output_file):
    """
    将两个WAV文件拼接成一个新的WAV文件
    
    参数:
        input_file1: 第一个输入WAV文件的路径
        input_file2: 第二个输入WAV文件的路径
        output_file: 输出WAV文件的路径
    """
    try:
        # 打开第一个WAV文件
        with wave.open(input_file1, 'rb') as wav1:
            # 获取第一个文件的参数
            params1 = wav1.getparams()
            frames1 = wav1.readframes(wav1.getnframes())
            
            # 打开第二个WAV文件
            with wave.open(input_file2, 'rb') as wav2:
                # 获取第二个文件的参数
                params2 = wav2.getparams()
                frames2 = wav2.readframes(wav2.getnframes())

                # # 检查两个文件的参数是否匹配
                # if (params1.nchannels != params2.nchannels or 
                #     params1.sampwidth != params2.sampwidth or 
                #     params1.framerate != params2.framerate):
                #     raise ValueError("两个WAV文件的音频参数不匹配")

                # 创建输出WAV文件
                with wave.open(output_file, 'wb') as output:
                    # 设置输出文件的参数
                    output.setparams(params2)
                    # 写入两个文件的音频数据
                    output.writeframes(frames1)
                    output.writeframes(frames2)
                    
        print(f"文件拼接成功! 已保存为: {output_file}")
        return True
        
    except wave.Error as e:
        print(f"WAV文件处理错误: {str(e)}")
        return False
    except ValueError as e:
        print(f"参数错误: {str(e)}")
        return False
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file1 = "/home/leon/agent/AISHELL_dataset/test/insert/S_R004S04C01_agent_added_fixed/base_add.wav"
    input_file2 = "/home/leon/agent/AISHELL_dataset/nihaojiaojiao.wav"
    output_file = "combined_output2.wav"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file1):
        print(f"错误: 文件 {input_file1} 不存在")
    elif not os.path.exists(input_file2):
        print(f"错误: 文件 {input_file2} 不存在")
    else:
        # 执行文件拼接
        concatenate_wav_files(input_file1, input_file2, output_file)