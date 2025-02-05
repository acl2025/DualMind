import os
from textgrid import TextGrid, IntervalTier, Interval
import time

# 使用Kimi Chat的API
from openai import OpenAI

client = OpenAI(
    api_key="sk-4Stc5dY49tGPqqcvMktNLDjeRpAlougEyKwX42Dlns6x4XiX",  # 请在此处替换为你从Kimi开放平台申请的API Key
    base_url="https://api.moonshot.cn/v1",
)

# 定义输入和输出文件夹路径
input_dir = "/home/leon/agent/AISHELL_dataset/train_L/TextGrid"
output_dir = "/home/leon/agent/AISHELL_dataset/train_L/TextGrid_agent_added"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入文件夹中的所有TextGrid文件
for filename in os.listdir(input_dir):
    if filename.endswith(".TextGrid"):
        print(f"正在处理文件 {filename}")
        input_file = os.path.join(input_dir, filename)
        file_prefix = os.path.splitext(filename)[0]
        rttm_filename = f"{file_prefix}.rttm"
        rttm_file = os.path.join(input_dir, rttm_filename)
        output_filename = f"{file_prefix}_agent_added.TextGrid"
        output_file = os.path.join(output_dir, output_filename)
        output_rttm_filename = f"{file_prefix}_agent_added.rttm"
        output_rttm_file = os.path.join(output_dir, output_rttm_filename)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file) and os.path.exists(output_rttm_file):
            print(f"输出文件 {output_filename} 和 {output_rttm_filename} 已存在，跳过处理")
            continue
        
        # 检查RTTM文件是否存在
        if not os.path.exists(rttm_file):
            print(f"RTTM文件 {rttm_filename} 未找到，跳过文件 {filename}")
            continue

        # 读取RTTM文件，提取发言段
        segments = []
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
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
                        'speaker_id': speaker_id
                    })

        # 按开始时间排序
        segments.sort(key=lambda x: x['start_time'])

        # 读取TextGrid文件
        tg = TextGrid()
        tg.read(input_file)

        # === 修改部分：统计每个说话者的文本总字数 ===
        speaker_text_lengths = {}
        for tier in tg.tiers:
            if isinstance(tier, IntervalTier):
                speaker_name = tier.name.strip()
                total_length = 0
                for interval in tier.intervals:
                    text = interval.mark.strip()
                    # 跳过空文本和以“你好交交”开头的文本
                    if text and not text.startswith("你好交交"):
                        total_length += len(text)
                speaker_text_lengths[speaker_name] = total_length

        if not speaker_text_lengths:
            print(f"在文件 {filename} 中，未检测到任何说话者的文本内容。")
            continue

        # 找出文本总字数最多的说话者
        most_active_speaker = max(speaker_text_lengths, key=speaker_text_lengths.get)
        print(f"文件 {filename} 中，文本总字数最多的说话者是 {most_active_speaker}")

        # 找到对应的IntervalTier
        target_tier = None
        for tier in tg.tiers:
            if isinstance(tier, IntervalTier) and tier.name.strip() == most_active_speaker:
                target_tier = tier
                break

        if not target_tier:
            print(f"在文件 {filename} 中，未找到与说话者 {most_active_speaker} 对应的Tier。")
            continue

        # 定义一个函数来收集会议内容
        def collect_meeting_content(start_time, end_time):
            collected_text = ''
            # 遍历所有的Tier
            for tier in tg.tiers:
                if isinstance(tier, IntervalTier):
                    # 查找在该Tier中，与时间窗口重叠的interval
                    for interval in tier.intervals:
                        interval_start = interval.minTime
                        interval_end = interval.maxTime
                        # 检查interval是否在时间窗口内
                        if interval_end <= start_time:
                            continue
                        if interval_start >= end_time:
                            break
                        # 重叠部分
                        overlap_start = max(interval_start, start_time)
                        overlap_end = min(interval_end, end_time)
                        if overlap_start < overlap_end:
                            if interval.mark.strip():
                                # 跳过以“你好交交”开头的间隔
                                if interval.mark.strip().startswith("你好交交"):
                                    continue
                                collected_text += interval.mark.strip()
            return collected_text

        # 获取会议的总时长
        meeting_duration = tg.maxTime

        # 设置时间间隔
        interval_duration = 500  # 每500秒生成一次唤醒问题

        # 初始化变量
        new_segments = []

        # 计算所有的时间点
        time_points = list(range(interval_duration, int(meeting_duration) + interval_duration, interval_duration))

        # 定义提问类型，循环使用
        question_types = ['type1', 'type2']  # 两种类型轮流使用

        # 定义一个极小的正数epsilon
        epsilon = 1e-10

        for idx, current_time in enumerate(time_points):
            start_time = max(0, current_time - interval_duration)
            end_time = current_time

            # 如果结束时间超过会议时长，调整为会议结束时间
            if end_time > tg.maxTime:
                print(f"结束时间 {end_time} 超过了会议总时长 {tg.maxTime}，将其调整为会议结束时间。")
                end_time = tg.maxTime

            # 收集会议内容
            collected_text = collect_meeting_content(start_time, end_time)
            if not collected_text:
                print(f"在文件 {filename} 中，时间点 {current_time} 没有找到足够的文本内容。")
                continue
            elif len(collected_text) < 600:
                print(f"在文件 {filename} 中，时间点 {current_time} 文本内容不足600个字符，将使用可用的文本。")

            # 根据问题类型设置Prompt
            q_type = question_types[idx % len(question_types)]
            if q_type == 'type1':
                prompt = f"""请根据以下会议内容生成一句让呼出会议助手交交提出的问题，请注意只需要输出结果且一定要开局要说你好交交且该针对性或介绍问题内容要简单，输出的总字数在40字以内：

### 会议内容示例 ###
这就是我觉得就是怎么去把我们物业这一块儿做的更好。我这边搜集的几个问题就是啥嘞
第一个就是说很多业主就反映啥嘞说，就是对外来这个车辆进入咱们小区。监管力度不够。
咱们就不是一个问题一个问题来吧，是吧，一个问题一个问题去讨论，解决这个问题的就去讨论下一个，解决一个问题咱们讨论下一个问题，
啊第一个就是车辆问题
很多业主都不满意。说我的车为什么别的车在这儿停着嘞，然后因为这个事儿还产生很多的一个纠纷
<%>而我的车位这儿我让你走，你也不走，打电话你也不来。所以这是确实有业主给咱们物业啊投这个组上，提出这个要求。
嗯，说我们这一块儿的话，在在管理上面，在监督上面，或者说就是存在咱们工作的一个疏忽。
对这个就是就是这个问题，一个车位就是一个车，很简单的例子在哪里，就是说比如说您是咱们业主，您是一个车位，
然后的话您可以对外去把这个车位租出去。对吧。
你可能诶。租出去两三辆车都可以随随便便进咱们小区是吧，这个地方的话都可以随便停。
这样对于我们管理的话，那绝对是一种错误的选择。
咱们就说一个车一个车位，这是肯定的。是吧，
现在小区也是这样，不是说每户都有车，所以说车位的话是吧，车位是有限的
这个就是说还是说到咱们怎么去管理，怎么去监督。
对对对
就是绿化部门的事儿对吧，咱们把绿化部门儿这个工作做好以后，人家在日常当中工作当中真正。是吧能够把工作做好的话，这些问题都是可以得到解决。
那就把主管撤了吧。换一个。那就是领导的问题，对不对，那领导出问题怎么办
###

### 示例输出 ###
你好交交, 针对外来车辆监管不足问题，你建议哪些改进物业管理措施的基本方法？
###
    
### 需要处理的会议内容记录 ###
{collected_text}
###
"""
            else:
                prompt = f"""请根据以下会议内容生成一句让呼出会议助手交交提出的问题，要求问题是上文关联型，例如“之前会议中提到的上个季度的产量是多少？”，请在输出中一定要是“之前会议中提到XXX”或“基于之前我们讨论的内容”。请注意只需要输出结果且一定要开局说你好交交,输出的总字数在80字以上：

### 会议内容示例 ###
这就是我觉得就是怎么去把我们物业这一块儿做的更好。我这边搜集的几个问题就是啥嘞
第一个就是说很多业主就反映啥嘞说，就是对外来这个车辆进入咱们小区。监管力度不够。
咱们就不是一个问题一个问题来吧，是吧，一个问题一个问题去讨论，解决这个问题的就去讨论下一个，解决一个问题咱们讨论下一个问题，
啊第一个就是车辆问题
很多业主都不满意。说我的车为什么别的车在这儿停着嘞，然后因为这个事儿还产生很多的一个纠纷
<%>而我的车位这儿我让你走，你也不走，打电话你也不来。所以这是确实有业主给咱们物业啊投这个组上，提出这个要求。
嗯，说我们这一块儿的话，在在管理上面，在监督上面，或者说就是存在咱们工作的一个疏忽。
对这个就是就是这个问题，一个车位就是一个车，很简单的例子在哪里，就是说比如说您是咱们业主，您是一个车位，
然后的话您可以对外去把这个车位租出去。对吧。
你可能诶。租出去两三辆车都可以随随便便进咱们小区是吧，这个地方的话都可以随便停。
这样对于我们管理的话，那绝对是一种错误的选择。
咱们就说一个车一个车位，这是肯定的。是吧，
现在小区也是这样，不是说每户都有车，所以说车位的话是吧，车位是有限的
这个就是说还是说到咱们怎么去管理，怎么去监督。
对对对
就是绿化部门的事儿对吧，咱们把绿化部门儿这个工作做好以后，人家在日常当中工作当中真正。是吧能够把工作做好的话，这些问题都是可以得到解决。
那就把主管撤了吧。换一个。那就是领导的问题，对不对，那领导出问题怎么办
###

### 示例输出1 ###
你好交交，基于之前我们讨论的内容，关于小区外来车辆管理这个问题，你觉得我们应该如何改进目前的车位租赁管理制度，以确保每个车位只对应一辆车，同时又能满足业主的合理需求呢？
###
    
### 示例输出2 ###
你好交交，之前会议中提到了物业在车辆管理方面存在监管不足的问题，特别是业主反映的外来车辆占用问题。我想请问一下，物业部门在车辆管理方面存在监管不严具体表现在哪些方面呢？
###
    
### 需要处理的会议内容记录 ###
{collected_text}
###
"""
            
            print(f"-------时间点 {current_time} 的prompt: ", prompt)
            time.sleep(1)  # 避免API请求过于频繁

            # 调用Kimi Chat API，生成问题
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {
                        "role": "system",
                        "content": "你是人工智能助手，你擅长根据会议内容生成相关的问题。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            # 获取生成的问题
            response = completion.choices[0].message.content.strip()
            generated_question = response

            print(f"-------时间点 {current_time} 的response: ", response)
            time.sleep(1)  # 避免API请求过于频繁

            # 确保生成的问题以“你好交交”开头
            if not generated_question.startswith("你好交交"):
                generated_question = "你好交交, " + generated_question

            # 获取插入位置
            insert_time = end_time
            
            
            # 如果插入时间超过会议时长，调整为会议结束时间
            if insert_time > tg.maxTime:
                print(f"插入时间 {insert_time} 超过了会议总时长 {tg.maxTime}，将其调整为会议结束时间。")
                insert_time = tg.maxTime

            # === 新增部分：检查冲突，调整插入时间 ===
            # 如果插入时间点上存在非空的间隔，需要调整插入时间到冲突间隔的结束时间
            def adjust_insert_time(tier, time_point):
                intervals = tier.intervals
                for interval in intervals:
                    if interval.minTime <= time_point < interval.maxTime:
                        return interval.maxTime
                return time_point

            # 调整插入时间，避免与非空间隔冲突
            adjusted_insert_time = adjust_insert_time(target_tier, insert_time)

            # 如果调整后的插入时间发生了变化，提示
            if adjusted_insert_time != insert_time:
                print(f"插入时间 {insert_time} 与非空间隔冲突，调整为 {adjusted_insert_time}")
                insert_time = adjusted_insert_time

            # 在 target_tier 中插入零时长间隔
            intervals = target_tier.intervals
            new_intervals = []
            inserted = False

            # 遍历现有的间隔，保持时间轴连续
            for i, interval in enumerate(intervals):
                # 如果还没有插入，且插入时间在当前间隔之前
                if not inserted and insert_time < interval.minTime:
                    # 插入新的零时长间隔
                    new_interval = Interval(insert_time, insert_time + epsilon, generated_question)
                    new_interval.minTime = insert_time
                    new_interval.maxTime = insert_time  # 手动设置为零时长
                    new_intervals.append(new_interval)
                    new_intervals.append(interval)
                    inserted = True
                elif not inserted and interval.minTime <= insert_time <= interval.maxTime:
                    # 插入时间在当前间隔内或边界上，需要处理
                    if interval.minTime == insert_time == interval.maxTime:
                        # 间隔为零时长
                        new_interval = Interval(insert_time, insert_time + epsilon, generated_question)
                        new_interval.minTime = insert_time
                        new_interval.maxTime = insert_time
                        new_intervals.append(new_interval)
                        new_intervals.append(interval)
                        inserted = True
                    elif interval.minTime == insert_time:
                        # 插入在当前间隔开始
                        new_interval = Interval(insert_time, insert_time + epsilon, generated_question)
                        new_interval.minTime = insert_time
                        new_interval.maxTime = insert_time
                        new_intervals.append(new_interval)
                        new_intervals.append(interval)
                        inserted = True
                    elif interval.maxTime == insert_time:
                        # 插入在当前间隔结束
                        new_intervals.append(interval)
                        new_interval = Interval(insert_time, insert_time + epsilon, generated_question)
                        new_interval.minTime = insert_time
                        new_interval.maxTime = insert_time
                        new_intervals.append(new_interval)
                        inserted = True
                    else:
                        # 插入时间在间隔内部，且之前已调整过，不应出现此情况
                        new_intervals.append(interval)
                else:
                    new_intervals.append(interval)

            if not inserted:
                # 如果插入时间在所有间隔之后，插入到末尾
                new_interval = Interval(insert_time, insert_time + epsilon, generated_question)
                new_interval.minTime = insert_time
                new_interval.maxTime = insert_time
                new_intervals.append(new_interval)

            # 更新 target_tier 的 intervals
            target_tier.intervals = new_intervals

            # 更新RTTM数据，添加新问题的记录，持续时间为0
            new_segment = {
                'start_time': insert_time,
                'duration': 0.0,  # 持续时间为0
                'speaker_id': most_active_speaker  # 插入到发言次数最多的说话者
            }
            new_segments.append(new_segment)

        # 将新的segments添加到原有segments
        segments.extend(new_segments)

        # 按开始时间排序
        segments.sort(key=lambda x: x['start_time'])

        # 将修改后的TextGrid写入输出文件
        tg.write(output_file)

        # 在写入 RTTM 文件之前，统一所有 segment 的格式
        formatted_segments = []
        for segment in segments:
            formatted_segment = {
                'start_time': segment['start_time'],
                'duration': segment['duration'],
                'speaker_id': segment['speaker_id']
            }
            formatted_segments.append(formatted_segment)

        # 按开始时间排序
        formatted_segments.sort(key=lambda x: x['start_time'])

        # 生成修改后的RTTM文件
        with open(output_rttm_file, 'w') as f:
            for segment in formatted_segments:
                f.write(f"SPEAKER {file_prefix} 1 {segment['start_time']:.10f} {segment['duration']:.10f} <NA> <NA> {segment['speaker_id']} <NA> <NA>\n")

        print(f"已成功将生成的问题插入到 {output_file}，并生成了新的RTTM文件 {output_rttm_filename}")