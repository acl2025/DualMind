# -*- coding: utf-8 -*-
import os
import re
import json

def process_files(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('_agent_added.txt'):
            txt_file_path = os.path.join(input_dir, filename)
            jsonl_file_path = os.path.join(output_dir, filename.replace('.txt', '.jsonl'))

            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式解析文件内容
            pattern = r'\[Agent\] 最终接收到问题: (.*?), \d+\.\d+'
            question_matches = list(re.finditer(pattern, content, re.DOTALL))

            data_list = []

            for idx, match in enumerate(question_matches):
                data = {}

                question = match.group(1).strip()
                data['question'] = question

                # 获取当前问题段落的内容
                start_pos = match.end()
                if idx + 1 < len(question_matches):
                    end_pos = question_matches[idx + 1].start()
                else:
                    end_pos = len(content)
                question_content = content[start_pos:end_pos]

                # 提取 planner first judgetoken 的输出
                planner_first_judgetoken_match = re.search(r'First token received: (\d)', question_content)
                if planner_first_judgetoken_match:
                    planner_first_judgetoken = planner_first_judgetoken_match.group(1)
                    data['planner_first_judgetoken'] = planner_first_judgetoken

                    # 提取处理时间信息
                    time_info_match = re.search(r'Token count: .*?seconds.*?One token time: .*?seconds', question_content, re.DOTALL)
                    if time_info_match:
                        data['processing_time_info'] = time_info_match.group().strip()
                    else:
                        data['processing_time_info'] = ''

                    if planner_first_judgetoken == '1':
                        # Talker 模式
                        talker_match = re.search(r'talker输出：\s*(.*?)talker 输出结束', question_content, re.DOTALL)
                        if talker_match:
                            data['talker_output'] = talker_match.group(1).strip()
                        else:
                            data['talker_output'] = ''
                    elif planner_first_judgetoken == '0':
                        # planner 模式
                        planner_match = re.search(r'===planner 进一步输出开始===\s*(.*?)===planner 输出结束===', question_content, re.DOTALL)
                        if planner_match:
                            planner_content = planner_match.group(1)

                            # 提取第一步 planner 的规划
                            action_match = re.search(r'1\. 行动：(.*?)\n', planner_content)
                            data['planner_plan'] = action_match.group(1).strip() if action_match else ''

                            # 提取"提取的关键词"
                            keywords_match = re.search(r'2\. 行动输入：关键词：(.*?)\n', planner_content)
                            data['extracted_keywords'] = keywords_match.group(1).strip() if keywords_match else ''

                            # 提取"RAG used context"
                            context_match = re.search(r'###RAG used context:###\s*(.*?)###agent根据会议片段的输出开始：###\n', planner_content, re.DOTALL)
                            #context_match = re.search(r'###RAG used context:### (.*?)###End RAG used context:###\n', planner_content, re.DOTALL)
                            data['text_snippet'] = context_match.group(1).strip() if context_match else ''

                            # 提取"planner 输出结束"的上一行作为agent工具结果
                            agent_tools_match = re.search(r' ###agent根据会议片段的输出开始：###(.*?)###agent根据会议片段的输出结束###', planner_content, re.DOTALL)
                            data['agent_tool_result'] = agent_tools_match.group(1).strip() if agent_tools_match else ''
                            
                        else:
                            data['planner_plan'] = ''
                            data['extracted_keywords'] = ''
                            data['text_snippet'] = ''
                            data['agent_tool_result'] = ''
                else:
                    data['planner_first_judgetoken'] = ''
                    data['processing_time_info'] = ''

                data_list.append(data)

            # 将数据写入 jsonl 文件
            with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                for item in data_list:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

if __name__ == '__main__':
    input_directory = '/home/leon/agent/experiment_result/result_train_S_audio_segment_only_0103'
    output_directory = '/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103'
    process_files(input_directory, output_directory)
