from textgrid import TextGrid, IntervalTier, Interval
import logging
import traceback
import os

logging.basicConfig(level=logging.WARNING)

# 第一步：猴子补丁 IntervalTier 的 addInterval 方法
def flexible_addInterval(self, interval):
    if interval.minTime >= interval.maxTime:
        raise ValueError("Interval minTime must be less than maxTime")
    # 如果 interval 的 maxTime 超过了当前 tier 的 maxTime，则更新 tier 的 maxTime
    if interval.maxTime > self.maxTime:
        self.maxTime = interval.maxTime
    self.intervals.append(interval)

# 应用猴子补丁
IntervalTier.addInterval = flexible_addInterval

# 第二步：修改 fix_overlaps_in_textgrid 函数
def fix_overlaps_in_textgrid(textgrid_path, output_path):
    # 使用非严格模式读取 TextGrid 文件
    tg = TextGrid(strict=False)
    tg.read(textgrid_path)
    
    epsilon = 0.001  # 一个非常小的正数，防止间隔长度为零

    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            intervals = tier.intervals
            # 按开始时间排序
            intervals.sort(key=lambda x: x.minTime)
            fixed_intervals = []
            prev_maxTime = None
            for interval in intervals:
                is_agent_interval = '你好交交' in (interval.mark or '')
                # 对于包含“你好交交”的间隔
                if is_agent_interval:
                    # 将 xmax 设置为 xmin + epsilon，避免零长度
                    interval.minTime = max(interval.minTime, prev_maxTime or 0.0)
                    interval.maxTime = interval.minTime + epsilon
                    fixed_intervals.append(interval)
                    prev_maxTime = interval.maxTime
                    continue

                # 如果有前一个间隔
                if prev_maxTime is not None:
                    if interval.minTime < prev_maxTime:
                        logging.warning(f"Overlap for interval {interval.mark}: ({interval.minTime:.6f}, {interval.maxTime:.6f})")
                        # 调整当前间隔的 minTime，避免重叠
                        interval.minTime = prev_maxTime
                    # 检查调整后 minTime 是否大于等于 maxTime
                    if interval.minTime >= interval.maxTime:
                        # 调整 maxTime，使间隔长度为 epsilon
                        interval.maxTime = interval.minTime + epsilon
                        logging.warning(f"Adjusted maxTime for interval {interval.mark}: ({interval.minTime:.6f}, {interval.maxTime:.6f})")
                else:
                    # 确保间隔的 minTime 小于 maxTime
                    if interval.minTime >= interval.maxTime:
                        # 调整 maxTime，使间隔长度为 epsilon
                        interval.maxTime = interval.minTime + epsilon
                        logging.warning(f"Adjusted maxTime for interval {interval.mark}: ({interval.minTime:.6f}, {interval.maxTime:.6f})")

                fixed_intervals.append(interval)
                prev_maxTime = interval.maxTime
            # 更新间隔列表
            tier.intervals = fixed_intervals

    # 更新 TextGrid 的 maxTime 为所有间隔的最大 maxTime
    max_times = [interval.maxTime for tier in tg.tiers if isinstance(tier, IntervalTier) for interval in tier.intervals]
    if max_times:
        tg.maxTime = max(max_times)
    else:
        tg.maxTime = 0.0

    # 验证间隔
    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            for interval in tier.intervals:
                if interval.minTime > interval.maxTime:
                    logging.error(f"Invalid interval in tier {tier.name}: {interval.mark} ({interval.minTime}, {interval.maxTime})")
                if interval.minTime < 0 or interval.maxTime < 0:
                    logging.error(f"Negative time in interval in tier {tier.name}: {interval.mark} ({interval.minTime}, {interval.maxTime})")

    # 保存修复后的 TextGrid 文件
    try:
        tg.write(output_path)
        print(f"已修复并保存新的 TextGrid 文件：{output_path}")
    except Exception as e:
        print(f"Error saving TextGrid file {output_path}: {e}")
        traceback.print_exc()

# 第三步：保持 generate_rttm_from_textgrid 函数不变

def generate_rttm_from_textgrid(textgrid_path, output_path):
    # 读取修复后的 TextGrid 文件
    tg = TextGrid(strict=False)
    tg.read(textgrid_path)
    segments = []
    file_id = os.path.splitext(os.path.basename(textgrid_path))[0].replace("_fixed", "")
    channel = '1'  # 根据需要调整通道号

    for tier in tg.tiers:
        if isinstance(tier, IntervalTier):
            speaker_name = tier.name.strip()
            for interval in tier.intervals:
                # 跳过没有标记的间隔
                if interval.mark and interval.mark.strip() != '':
                    start_time = interval.minTime
                    duration = interval.maxTime - interval.minTime
                    if duration <= 0:
                        continue  # 跳过零长度和负长度的间隔
                    speaker_id = speaker_name
                    # 构建 RTTM 行
                    rttm_line = f"SPEAKER {file_id} {channel} {start_time:.4f} {duration:.4f} <NA> <NA> {speaker_id} <NA> <NA>"
                    segments.append(rttm_line)

    # 检查是否有段可写入 RTTM 文件
    if segments:
        # 保存生成的 RTTM 文件
        with open(output_path, 'w') as f:
            for line in segments:
                f.write(line + '\n')
        print(f"已生成新的 RTTM 文件：{output_path}")
    else:
        print(f"警告：未找到有效的段，未生成 RTTM 文件：{output_path}")

# 第四步：修改 batch_process_files 函数
def batch_process_files():
    # 定义输入和输出目录
    input_dir = "/home/leon/agent/AISHELL_dataset/train_S/TextGrid_agent_added"
    output_dir = "/home/leon/agent/AISHELL_dataset/train_S/TextGrid_agent_added_0interval"

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
                    # 生成与修复后的 TextGrid 文件对应的 RTTM 文件
                    generate_rttm_from_textgrid(textgrid_output, rttm_output)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错：{str(e)}")
                    traceback.print_exc()
            else:
                print(f"警告：找不到对应的 RTTM 文件：{rttm_input}")

if __name__ == "__main__":
    batch_process_files()