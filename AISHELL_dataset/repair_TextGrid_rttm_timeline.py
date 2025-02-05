from textgrid import TextGrid, IntervalTier, Interval
import os
from collections import Counter

def fix_overlaps_in_textgrid(textgrid_path, output_path):
    # 使用非严格模式读取 TextGrid 文件
    tg = TextGrid(strict=False)
    tg.read(textgrid_path)
    
    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            intervals = tier.intervals
            # 按开始时间排序
            intervals.sort(key=lambda x: x.minTime)
            fixed_intervals = []
            prev_interval = None
            for interval in intervals:
                if prev_interval is None:
                    fixed_intervals.append(interval)
                    prev_interval = interval
                else:
                    if interval.minTime < prev_interval.maxTime:
                        # 发现重叠，调整当前间隔的开始时间
                        interval.minTime = prev_interval.maxTime
                        if interval.minTime >= interval.maxTime:
                            # 如果间隔长度为零或负数，跳过该间隔
                            continue
                    fixed_intervals.append(interval)
                    prev_interval = interval
            # 更新间隔列表
            tier.intervals = fixed_intervals
        
    # 保存修复后的 TextGrid 文件
    tg.write(output_path)
    print(f"已修复并保存新的 TextGrid 文件：{output_path}")
    
def fix_overlaps_in_rttm(rttm_path, textgrid_path, output_path):
    # 首先读取 TextGrid 文件，建立时间段与说话人名称的对应关系
    tg = TextGrid(strict=False)
    tg.read(textgrid_path)
    speaker_intervals = []
    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            speaker_name = tier.name.strip()
            for interval in tier.intervals:
                if interval.mark.strip() != '':
                    speaker_intervals.append({
                        'minTime': interval.minTime,
                        'maxTime': interval.maxTime,
                        'speaker': speaker_name,
                        'text': interval.mark.strip()
                    })
    # 按开始时间排序
    speaker_intervals.sort(key=lambda x: x['minTime'])
    
    # 读取 RTTM 文件，提取段信息
    segments = []
    file_id = None
    channel = None
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                if file_id is None:
                    file_id = parts[1]
                    channel = parts[2]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                end_time = start_time + duration
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'speaker_id': speaker_id,
                    'line': line.strip(),
                    'parts': parts
                })
    
    # 更新 RTTM 段的说话者ID
    for segment in segments:
        matching_intervals = []
        for interval in speaker_intervals:
            # 检查是否有重叠
            overlap_start = max(segment['start_time'], interval['minTime'])
            overlap_end = min(segment['end_time'], interval['maxTime'])
            if overlap_start < overlap_end:
                matching_intervals.append(interval)
        if matching_intervals:
            # 优先选择讲话内容中包含"你好交交"的说话者
            agent_speaker = None
            for interval in matching_intervals:
                if '你好交交' in interval['text']:
                    agent_speaker = interval['speaker']
                    break  # 找到后立即退出循环
            if agent_speaker:
                segment['speaker_id'] = agent_speaker
            else:
                # 如果没有找到，则选择出现次数最多的说话者
                matching_speakers = [interval['speaker'] for interval in matching_intervals]
                speaker_counter = Counter(matching_speakers)
                most_common_speaker = speaker_counter.most_common(1)[0][0]
                segment['speaker_id'] = most_common_speaker
            # 更新 RTTM 行中的说话人名称
            segment['parts'][7] = segment['speaker_id']
            segment['line'] = ' '.join(segment['parts'])
        else:
            print(f"未找到匹配的说话人，段时间：{segment['start_time']} - {segment['end_time']}，保留原说话者：{segment['speaker_id']}")
    
    # 找出 TextGrid 中未在 RTTM 中出现的时间间隔
    unmatched_intervals = []
    for interval in speaker_intervals:
        matched = False
        for segment in segments:
            overlap_start = max(interval['minTime'], segment['start_time'])
            overlap_end = min(interval['maxTime'], segment['end_time'])
            if overlap_start < overlap_end:
                matched = True
                break
        if not matched:
            unmatched_intervals.append(interval)
    
    # 将未匹配的时间间隔添加为新的 RTTM 段
    for interval in unmatched_intervals:
        start_time = interval['minTime']
        duration = interval['maxTime'] - interval['minTime']
        speaker_id = interval['speaker']
        new_segment_line = f"SPEAKER {file_id} {channel} {start_time:.4f} {duration:.4f} <NA> <NA> {speaker_id} <NA> <NA>"
        segments.append({
            'start_time': start_time,
            'end_time': interval['maxTime'],
            'duration': duration,
            'speaker_id': speaker_id,
            'line': new_segment_line,
            'parts': new_segment_line.strip().split()
        })
    
    # 按开始时间排序
    segments.sort(key=lambda x: x['start_time'])
    
    # 修复 RTTM 段的重叠
    fixed_segments = []
    prev_segment = None
    for segment in segments:
        if prev_segment is None:
            fixed_segments.append(segment)
            prev_segment = segment
        else:
            if segment['start_time'] < prev_segment['end_time']:
                # 发现重叠，调整当前段的开始时间
                segment['start_time'] = prev_segment['end_time']
                segment['duration'] = segment['end_time'] - segment['start_time']
                if segment['duration'] <= 0:
                    # 如果持续时间为零或负数，跳过该段
                    continue
                # 更新 RTTM 行中的时间信息
                segment['parts'][3] = f"{segment['start_time']:.4f}"
                segment['parts'][4] = f"{segment['duration']:.4f}"
                segment['line'] = ' '.join(segment['parts'])
            fixed_segments.append(segment)
            prev_segment = segment
    
    # 保存修复后的 RTTM 文件
    with open(output_path, 'w') as f:
        for segment in fixed_segments:
            f.write(segment['line'] + '\n')
    print(f"已修复并保存新的 RTTM 文件：{output_path}")

def batch_process_files():
    # 定义输入和输出目录
    input_dir = "/home/leon/agent/AISHELL_dataset/train_S/TextGrid_agent_added"
    output_dir = "/home/leon/agent/AISHELL_dataset/train_S/TextGrid_agent_added_fixed"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有 TextGrid 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.TextGrid'):
            # 构建输入文件路径
            textgrid_input = os.path.join(input_dir, filename)
            rttm_input = os.path.join(input_dir, filename.replace('.TextGrid', '.rttm'))
            
            # 构建输出文件路径
            base_name = os.path.splitext(filename)[0]
            textgrid_output = os.path.join(output_dir, f"{base_name}_fixed.TextGrid")
            rttm_output = os.path.join(output_dir, f"{base_name}_fixed.rttm")
            
            # 检查对应的 RTTM 文件是否存在
            if os.path.exists(rttm_input):
                print(f"\n处理文件对：{filename}")
                try:
                    # 先处理 TextGrid 文件
                    fix_overlaps_in_textgrid(textgrid_input, textgrid_output)
                    # 然后处理 RTTM 文件
                    fix_overlaps_in_rttm(rttm_input, textgrid_output, rttm_output)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错：{str(e)}")
            else:
                print(f"警告：找不到对应的 RTTM 文件：{rttm_input}")

if __name__ == "__main__":
    batch_process_files()