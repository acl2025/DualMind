import os
import whisper
from glob import glob
import textgrid
import torch
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate
from spectralcluster import SpectralClusterer, RefinementOptions

# 设置要使用的GPU设备索引
device_index = 0  # 使用第一张显卡（索引从0开始），根据需要修改
torch.cuda.set_device(device_index)
device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

# 初始化Whisper模型，建议使用较小的模型以减少显存占用
model = whisper.load_model("large", device=device)

input_dir = "/home/leon/agent/AISHELL_dataset/test/wav"
output_dir = "/home/leon/agent/AISHELL_dataset/test/STT_by_whisper"
os.makedirs(output_dir, exist_ok=True)

# 获取所有FLAC文件的列表
flac_files = glob(os.path.join(input_dir, "*.flac"))

for flac_file in flac_files:
    base_name = os.path.splitext(os.path.basename(flac_file))[0]
    print(f"Processing {base_name}...")

    # 1. 使用Whisper进行语音识别
    print("Transcribing with Whisper...")
    result = model.transcribe(flac_file, language='zh', verbose=False)

    # 释放Whisper模型占用的显存
    torch.cuda.empty_cache()

    # 2. 使用Resemblyzer进行说话人分离
    print("Performing speaker diarization with Resemblyzer...")

    # 加载音频并预处理
    wav = preprocess_wav(flac_file)

    encoder = VoiceEncoder(device=device)

    # 使用较长的窗口和步长
    window_length = 10.0  # 秒
    hop_length = 5.0      # 秒

    window_length_samples = int(sampling_rate * window_length)
    hop_length_samples = int(sampling_rate * hop_length)

    total_samples = len(wav)
    n_frames = 1 + int((total_samples - window_length_samples) / hop_length_samples)

    frame_embeddings = []
    speaker_segments = []

    for i in range(n_frames):
        start_sample = i * hop_length_samples
        end_sample = start_sample + window_length_samples

        if end_sample > total_samples:
            end_sample = total_samples

        wave_segment = wav[start_sample:end_sample]
        embedding = encoder.embed_utterance(wave_segment)

        frame_embeddings.append(embedding)

        start_time = start_sample / sampling_rate
        end_time = end_sample / sampling_rate
        speaker_segments.append({
            'start': start_time,
            'end': end_time,
            'embedding': embedding
        })

        # 打印进度
        if i % 10 == 0:
            print(f"Processed {i}/{n_frames} frames")

    # 处理最后不足一个窗口的部分
    if end_sample < total_samples:
        wave_segment = wav[end_sample:]
        if len(wave_segment) > sampling_rate:  # 至少1秒
            embedding = encoder.embed_utterance(wave_segment)
            frame_embeddings.append(embedding)
            start_time = end_sample / sampling_rate
            end_time = total_samples / sampling_rate
            speaker_segments.append({
                'start': start_time,
                'end': end_time,
                'embedding': embedding
            })

    # 释放Resemblyzer占用的内存
    del encoder
    torch.cuda.empty_cache()

    # 准备嵌入向量用于聚类
    embedding_vectors = np.array(frame_embeddings)

    # 使用Spectral Clustering进行聚类
    refinement_options = RefinementOptions(
        gaussian_blur_sigma=1,
        p_percentile=0.90
    )

    # 如果已知说话人数量为5，可以设置 min_clusters 和 max_clusters 为5
    clusterer = SpectralClusterer(
        min_clusters=5,
        max_clusters=5,
        refinement_options=refinement_options
    )

    labels = clusterer.predict(embedding_vectors)

    # 为每个说话人分段添加聚类标签
    for i, label in enumerate(labels):
        speaker_segments[i]['speaker'] = f"Speaker_{label}"

    # 添加创建RTTM文件的部分
    # 创建RTTM文件
    rttm_file = os.path.join(output_dir, f"{base_name}.rttm")
    with open(rttm_file, "w") as f:
        for seg in speaker_segments:
            speaker = seg['speaker']
            start_time = seg['start']
            duration = seg['end'] - seg['start']
            f.write(f"SPEAKER {base_name} 1 {start_time:.4f} {duration:.4f} <NA> <NA> {speaker} <NA> <NA>\n")

    # 创建TextGrid文件
    tg = textgrid.TextGrid()
    tg.minTime = 0

    # 调整tg.maxTime
    # 获取说话人分段和识别文本段的最大结束时间
    max_speaker_time = speaker_segments[-1]['end']
    max_segment_time = result['segments'][-1]['end']
    tg.maxTime = max(max_speaker_time, max_segment_time)

    # 获取所有说话人
    speakers = set(seg['speaker'] for seg in speaker_segments)
    speaker_tiers = {speaker: textgrid.IntervalTier(name=speaker, minTime=0, maxTime=tg.maxTime) for speaker in speakers}
    for tier in speaker_tiers.values():
        tg.append(tier)

    # 为每个识别的文本段分配说话人
    for segment in result['segments']:
        ts_start = segment['start']
        ts_end = segment['end']
        text = segment['text']

        # 找到与文本段重叠的说话人分段
        overlapping_speakers = []
        for seg in speaker_segments:
            overlap_start = max(ts_start, seg['start'])
            overlap_end = min(ts_end, seg['end'])
            if overlap_start < overlap_end:
                overlapping_speakers.append(seg['speaker'])

        if overlapping_speakers:
            # 选取出现次数最多的说话人
            speaker = max(set(overlapping_speakers), key=overlapping_speakers.count)
        else:
            speaker = "Unknown"
            if speaker not in speaker_tiers:
                speaker_tiers[speaker] = textgrid.IntervalTier(name=speaker, minTime=0, maxTime=tg.maxTime)
                tg.append(speaker_tiers[speaker])

        # 确保区间的结束时间不超过tg.maxTime
        if ts_end > tg.maxTime:
            ts_end = tg.maxTime

        # **修改开始：舍入时间戳，避免浮点数精度问题**
        ts_start = round(ts_start, 4)
        ts_end = round(ts_end, 4)
        # **修改结束**

        # **修改开始：避免区间重叠**
        # 检查当前tier的最后一个区间的结束时间，如果与当前区间的开始时间有重叠，调整当前区间的开始时间
        tier_intervals = speaker_tiers[speaker].intervals
        if tier_intervals and ts_start < tier_intervals[-1].maxTime:
            ts_start = tier_intervals[-1].maxTime

        # 如果调整后开始时间不小于结束时间，则跳过该区间
        if ts_start >= ts_end:
            continue
        # **修改结束**

        interval = textgrid.Interval(ts_start, ts_end, text.strip())
        speaker_tiers[speaker].addInterval(interval)

    # 保存TextGrid文件
    tg_file = os.path.join(output_dir, f"{base_name}.TextGrid")
    tg.write(tg_file)

    print(f"Finished processing {base_name}.")

print("All files have been processed.")
