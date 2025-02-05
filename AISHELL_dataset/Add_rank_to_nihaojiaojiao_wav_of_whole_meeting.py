#有问题，未实现
import os
import re
from praatio import textgrid

# 设置路径
insert_jiaojiao_dir = '/home/leon/agent/AISHELL_dataset/test/insert_jiaojiao/'
textgrid_dir = '/home/leon/agent/AISHELL_dataset/test/TextGrid_agent_added_fixed_nihaojiaojiao/'

# 初始化整体出现次数索引
overall_occurrence_index = 0

# 遍历insert_jiaojiao目录下的文件夹
for dir_name in os.listdir(insert_jiaojiao_dir):
    dir_path = os.path.join(insert_jiaojiao_dir, dir_name)
    if not os.path.isdir(dir_path):
        continue

    # 构建TextGrid文件的路径
    textgrid_filename = os.path.join(textgrid_dir, dir_name + '.TextGrid')
    if not os.path.exists(textgrid_filename):
        print(f"TextGrid文件 {textgrid_filename} 不存在。")
        continue

    # 解析TextGrid文件
    tg = textgrid.openTextgrid(textgrid_filename, includeEmptyIntervals=True)

    # 初始化每个说话者的出现次数字典
    speaker_occurrence_dict = {}

    # 遍历每个层（说话者）
    for tier in tg.tiers:
        if not isinstance(tier, textgrid.IntervalTier):
            continue  # 如果不是IntervalTier，则跳过

        tier_name = tier.name  # 获取层的名称，即说话者的名字

        # 初始化说话者的出现次数Z
        Z = 0
        # 遍历每个区间
        for entry in tier.entries:
            start_time, end_time, label = entry
            if "你好交交" in label:
                # 获取说话者姓名
                speaker_name = tier_name
                # 更新说话者的出现次数Z
                if speaker_name not in speaker_occurrence_dict:
                    speaker_occurrence_dict[speaker_name] = 0
                else:
                    speaker_occurrence_dict[speaker_name] += 1
                Z = speaker_occurrence_dict[speaker_name]
                # 获取整体出现次数Y
                Y = overall_occurrence_index
                overall_occurrence_index += 1
                # 构建旧的文件名
                old_filename_pattern = f'out_{speaker_name}_*.wav'
                speaker_files = sorted([f for f in os.listdir(dir_path) if re.match(rf'out_{re.escape(speaker_name)}_\d+\.wav', f)])
                if Z < len(speaker_files):
                    old_filename = speaker_files[Z]
                    # 构建新的文件名
                    new_filename = f'out_{speaker_name}_{Z}_{Y}.wav'
                    old_file_path = os.path.join(dir_path, old_filename)
                    new_file_path = os.path.join(dir_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"已重命名 {old_file_path} 为 {new_file_path}")
                else:
                    print(f"在 {dir_path} 中没有找到匹配的音频文件，针对说话者 {speaker_name} 的第 {Z} 次出现。")
