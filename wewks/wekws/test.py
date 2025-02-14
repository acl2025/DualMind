import librosa
from wekws.bin.stream_kws_ctc import KeyWordSpotter
import datetime
import numpy as np

kws_xiaowen = KeyWordSpotter(
    ckpt_path='examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
    config_path='examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
    token_path='examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
    lexicon_path='examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
    threshold=0.0001,
    min_frames=3,
    max_frames=3000,
    interval_frames=30,
    score_beam=10,
    path_beam=40,
    gpu=-1,
    is_jit_model=False,
)

kws_xiaowen.set_keywords("交交")

def detection(audio, kw):
    kws = kws_xiaowen
    
    if audio is None:
        return "Input Error! Please enter one audio!"

    # 加载音频文件
    y, sr = librosa.load(audio, sr=16000)
    
    # 计算10秒对应的采样点数
    segment_length = 10 * sr  # 10秒的采样点数
    overlap = 1 * sr  # 1秒的重叠，防止漏检
    
    results = []
    
    # 分段处理音频
    for start_idx in range(0, len(y), segment_length - overlap):
        # 重置检测器状态
        kws.reset_all()
        
        # 提取当前段的音频
        end_idx = min(start_idx + segment_length, len(y))
        segment = y[start_idx:end_idx]
        
        # 转换为字节格式
        segment_wav = (segment * (1 << 15)).astype("int16").tobytes()
        
        # 对当前段进行检测
        # 使用较小的窗口进行滑动检测
        interval = int(3 * 16000) * 2  # 3秒窗口
        step = int(1 * 16000) * 2      # 1秒步长
        
        for i in range(0, len(segment_wav), step):
            chunk_wav = segment_wav[i:min(i + interval, len(segment_wav))]
            result = kws.forward(chunk_wav)
            print(f"Processing segment {start_idx//sr}-{end_idx//sr}s, chunk {i//(2*16000)}-{min(i + interval, len(segment_wav))//(2*16000)}s")
            print("Result:", result)
            
            if 'state' in result and result['state'] == 1:
                # 调整时间戳以反映在整个音频中的位置
                absolute_start = start_idx/sr + result['start']
                absolute_end = start_idx/sr + result['end']
                
                detection_result = {
                    'keyword': result['keyword'],
                    'start': absolute_start,
                    'end': absolute_end,
                    'text': f'Activated: Detect {result["keyword"]} from {absolute_start:.2f} to {absolute_end:.2f} second.'
                }
                results.append(detection_result)
                print(detection_result['text'])
    
    # 返回所有检测结果
    if results:
        return "\n".join([r['text'] for r in results])
    return "Deactivated."

# 测试代码
input_path = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/L_R004S02C01_agent_added_fixed/base_add.wav'
kw_input = 'huiyizhushou'

print(detection(audio=input_path, kw=kw_input))