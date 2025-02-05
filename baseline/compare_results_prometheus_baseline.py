import os
import json
import re
import torch

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

# ---------------------------------
#  加载 PrometheusEval 模型
# ---------------------------------
def load_model():
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", gpu_memory_utilization=0.9, max_model_len=16384)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    return judge

# ---------------------------------
#  清洗助手回答中的无关内容
# ---------------------------------
def clean_assistant_answer(answer: str) -> str:
    """
    使用正则删除诸如 system\n...assistant\n 等内容，并去除首尾空格
    """
    answer = re.sub(r"system\s*\n.*?\nuser\s*\n.*?\nassistant\s*\n", "", answer, flags=re.DOTALL)
    answer = answer.strip()
    return answer

# ---------------------------------
#  主流程：处理单个实验结果文件
# ---------------------------------
def process_experiment_file(exp_file_path, ground_truth_data):
    """
    1. 遍历实验结果的每个问题
    2. 分类简单/复杂
    3. 匹配 ground truth
    4. 调用 PrometheusEval 评分
    5. 实时写入 .jsonl 和 .txt 文件
    6. 每处理一个问题后，重置模型
    """
    with open(exp_file_path, 'r', encoding='utf-8') as f:
        exp_results = [json.loads(line) for line in f]
    print(f"正在处理实验结果文件: {exp_file_path}")

    # 获取文件名
    file_name = os.path.basename(exp_file_path).replace('.jsonl', '')
    # output_jsonl_path = f"/home/leon/agent/benchmark/result_prometheus_eval_train_S_0114/result_prometheus_eval_{file_name}.jsonl"
    # output_txt_path   = f"/home/leon/agent/benchmark/result_prometheus_eval_train_S_0114/result_prometheus_eval_{file_name}.txt"

    #output_jsonl_path = f"/home/leon/agent/baseline/result_prometheus_eval_Llama-2-7b-chat-hf_baseline_0115/result_prometheus_eval_Llama-2-7b-chat-hf_baseline_{file_name}.jsonl"
    #output_txt_path   = f"/home/leon/agent/baseline/result_prometheus_eval_Llama-2-7b-chat-hf_baseline_0115/result_prometheus_eval_Llama-2-7b-chat-hf_baseline_{file_name}.txt"
    
    # output_jsonl_path = f"/home/leon/agent/baseline/result_prometheus_eval_Llama-2-13b-chat-hf_baseline_0115/result_prometheus_eval_Llama-2-13b-chat-hf_baseline_{file_name}.jsonl"
    # output_txt_path   = f"/home/leon/agent/baseline/result_prometheus_eval_Llama-2-13b-chat-hf_baseline_0115/result_prometheus_eval_Llama-2-13b-chat-hf_baseline_{file_name}.txt"
    
    # output_jsonl_path = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-7B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2.5-7B-Instruct_baseline_{file_name}.jsonl"
    # output_txt_path   = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-7B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2.5-7B-Instruct_baseline_{file_name}.txt"
    
    # output_jsonl_path = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-14B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2.5-14B-Instruct_baseline_{file_name}.jsonl"
    # output_txt_path   = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-14B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2.5-14B-Instruct_baseline_{file_name}.txt"
    
    output_jsonl_path = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2Audio-7B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2Audio-7B-Instruct_baseline_{file_name}.jsonl"
    output_txt_path   = f"/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2Audio-7B-Instruct_baseline_0115/result_prometheus_eval_train_S_Qwen2Audio-7B-Instruct_baseline_{file_name}.txt"
    
    # 读取已处理问题，避免重复
    if os.path.exists(output_jsonl_path):
        processed_questions = set()
        with open(output_jsonl_path, 'r', encoding='utf-8') as f_out_jsonl:
            for line in f_out_jsonl:
                try:
                    record = json.loads(line)
                    if 'question' in record:
                        processed_questions.add(record['question'])
                except:
                    pass
    else:
        processed_questions = set()

    unprocessed_questions = []
    for exp_result in exp_results:
        question = exp_result.get('question', '').strip()
        if not question:
            continue
        if question in processed_questions:
            continue
        unprocessed_questions.append(exp_result)

    if not unprocessed_questions:
        print(f"文件 {exp_file_path} 中的所有问题都已处理完毕。")
        return

    print("发现未处理的问题，正在加载模型...")
    judge = load_model()

    # 保证输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out_jsonl, \
         open(output_txt_path, 'a', encoding='utf-8') as f_out_txt:

        for exp_result in unprocessed_questions:
            question = exp_result.get('question', '').strip()
            question_len = len(question)
            question_type = 'simple' if question_len <= 40 else 'complex'

            # 取实验结果的助手回答
            assistant_answer = get_assistant_answer(exp_result)
            # 去掉可能的 "Token count:" 等提示
            if assistant_answer.startswith("Token count:"):
                assistant_answer = re.sub(
                    r"Token count: \d+, Total time: \d+\.\d+ seconds, One token time: \d+\.\d+ seconds",
                    "",
                    assistant_answer
                ).strip()
            # 清洗无关内容
            assistant_answer = clean_assistant_answer(assistant_answer)

            # 使用同时比较 question + answer 的匹配函数
            matched_gt = find_matching_ground_truth(
                exp_question=question,
                exp_answer=assistant_answer,
                question_type=question_type,
                ground_truth_data=ground_truth_data
            )
            if not matched_gt:
                continue

            # 取到 ground truth
            gt_question = matched_gt.get('question', '').strip()
            gt_answer   = matched_gt.get('answer', '').strip()

            # 使用 PrometheusEval 进行评分
            instruction = gt_question
            response = assistant_answer
            reference_answer = gt_answer

            feedback, score = run_prometheus_eval(judge, instruction, response, reference_answer)

            # 写入 jsonl
            output_record = {
                'question': question,
                'instruction': instruction,
                'reference_answer': reference_answer,
                'assistant_answer': response,
                'feedback': feedback,
                'score': score
            }
            f_out_jsonl.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            f_out_jsonl.flush()

            # 写入 txt
            f_out_txt.write(f"Question: {question}\n")
            f_out_txt.write(f"Instruction:\n{instruction}\n\n")
            f_out_txt.write(f"Reference Answer:\n{reference_answer}\n\n")
            f_out_txt.write(f"Assistant Answer:\n{assistant_answer}\n\n")
            f_out_txt.write(f"Feedback:\n{feedback}\n\n")
            f_out_txt.write(f"Score: {score}\n")
            f_out_txt.write("-" * 50 + "\n")
            f_out_txt.flush()

            processed_questions.add(question)

    print(f"文件 {exp_file_path} 处理完毕。")
    return

# ---------------------------------
#  使用 PrometheusEval 进行评分
# ---------------------------------
def run_prometheus_eval(judge, instruction, response, reference_answer):
    """
    使用 PrometheusEval 对 assistant 的回答进行评分
    """
    # 定义评分标准
    # rubric_data = {
    #     "criteria": "会议助手回答的质量，包括正确性、相关性、连贯性和有用性。",
    #     "score1_description": "会议助手回答不正确、不相关或无用。",
    #     "score2_description": "会议助手回答存在重大正确性或相关性问题。",
    #     "score3_description": "回答基本正确和相关，但存在一些问题。",
    #     "score4_description": "回答大多正确、相关和有用，可能存在细微问题。",
    #     "score5_description": "回答完全正确、高度相关、连贯且非常有用。"
    # }
    
    rubric_data = {
    "criteria": "会议助手回答的质量，基于以下五个维度的综合评估：事实正确性、满足用户需求、简洁度、结构性和完整性。",
    
    "score1_description": """
    - 事实正确性：回答包含严重的事实错误
    - 满足用户需求：完全没有回应用户的核心问题
    - 简洁度：内容冗长重复，表达混乱
    - 结构性：结构混乱，重点不清
    - 完整性：严重遗漏关键会议内容
    综合表现：回答不正确、不相关或无用。
    """,
    
    "score2_description": """
    - 事实正确性：回答存在明显的事实偏差
    - 满足用户需求：对用户需求的理解和回应存在重大偏差
    - 简洁度：表达较为冗长，存在较多重复内容
    - 结构性：结构不清晰，重点不突出
    - 完整性：遗漏较多重要会议内容
    综合表现：回答存在重大正确性或相关性问题。
    """,
    
    "score3_description": """
    - 事实正确性：回答基本准确，可能有小的事实偏差
    - 满足用户需求：基本回应了用户需求，但不够全面
    - 简洁度：表达基本简洁，个别地方可以优化
    - 结构性：结构基本清晰，重点基本突出
    - 完整性：覆盖了主要会议内容，可能遗漏次要信息
    综合表现：回答基本正确和相关，但存在一些问题。
    """,
    
    "score4_description": """
    - 事实正确性：回答准确，极少有误差
    - 满足用户需求：很好地满足了用户需求
    - 简洁度：表达简洁清晰，很少有冗余
    - 结构性：结构清晰，重点突出
    - 完整性：较全面地覆盖了相关会议内容
    综合表现：回答大多正确、相关和有用，可能存在细微问题。
    """,
    
    "score5_description": """
    - 事实正确性：回答完全准确无误
    - 满足用户需求：完全满足用户需求，且提供了额外的有价值信息
    - 简洁度：表达极其简洁，没有任何冗余
    - 结构性：结构非常清晰，重点非常突出
    - 完整性：完整覆盖所有相关会议内容，可能超越参考答案
    综合表现：回答完全正确、高度相关、连贯且非常有用。
    """
}
    
    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    feedback, score = judge.single_absolute_grade(
        instruction=instruction,
        response=response,
        reference_answer=reference_answer,
        rubric=score_rubric
    )
    return feedback, score

# ---------------------------------
#  同时根据实验结果的 question 和 answer 进行匹配
# ---------------------------------
def find_matching_ground_truth(exp_question, exp_answer, question_type,
                               ground_truth_data, similarity_threshold=0.1):
    """
    对 ground truth 数据进行匹配，找到与实验结果对应的条目
    """
    best_match = None
    best_sim = 0.0

    for gt in ground_truth_data:
        gt_question = gt.get('question', '')
        gt_answer   = gt.get('answer', '')

        # 分别计算 question 与 answer 的相似度
        question_sim = compute_similarity(exp_question, gt_question)
        answer_sim   = compute_similarity(exp_answer,   gt_answer)

        # 使用最大相似度作为匹配指标
        combined_sim = max(question_sim, answer_sim)

        if combined_sim > best_sim:
            best_sim = combined_sim
            best_match = gt

    # 若最终 best_sim 不足阈值，则返回 None
    if best_sim >= similarity_threshold:
        print(f"匹配成功: best_sim={best_sim:.2f}, best_match question={best_match.get('question')}") 
        return best_match
    else:
        return None

# ---------------------------------
#  简易相似度计算示例
# ---------------------------------
def compute_similarity(q1: str, q2: str) -> float:
    """
    最简单的字符集重叠度来衡量相似度，可换用更高级的方法
    """
    set1 = set(q1)
    set2 = set(q2)
    overlap = set1 & set2
    if max(len(set1), len(set2)) == 0:
        return 0.0
    return len(overlap) / max(len(set1), len(set2))

# ---------------------------------
#  获取助手答案
# ---------------------------------
def get_assistant_answer(exp_result: dict) -> str:
    """
    优先从 talker_output 获取；若无，再从 agent_tool_result 获取
    """
    assistant_answer = ''
    # if 'talker_output' in exp_result and exp_result['talker_output']:
    #     assistant_answer = exp_result['talker_output']
    # elif 'agent_tool_result' in exp_result and exp_result['agent_tool_result']:
    #     assistant_answer = exp_result['agent_tool_result']
    if 'answer' in exp_result and exp_result['answer']:
        assistant_answer = exp_result['answer']
    else:
        assistant_answer = 'none'

    # 清洗无关内容
    assistant_answer = clean_assistant_answer(assistant_answer)
    return assistant_answer

# ---------------------------------
#  脚本入口
# ---------------------------------
def main():
    # Ground Truth 和实验结果目录
    ground_truth_dir = "/home/leon/agent/AISHELL_dataset/train_S/ground_truth_train_S"
    #exp_results_dir = "/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103"
    #exp_results_dir = "/home/leon/agent/baseline/result_Llama-2-7b-chat-hf_baseline"
    #exp_results_dir = "/home/leon/agent/baseline/result_Llama-2-13b-chat-hf_baseline"
    #exp_results_dir = "/home/leon/agent/baseline/result_Qwen2.5-7B-Instruct_baseline"
    #exp_results_dir = "/home/leon/agent/baseline/result_Qwen2.5-14B-Instruct_baseline"
    exp_results_dir = "/home/leon/agent/baseline/result_Qwen2Audio-7B-Instruct_baseline"
    
    for root, dirs, files in os.walk(exp_results_dir):
        for file in files:
            if file.endswith('.jsonl'):
                exp_file_path = os.path.join(root, file)
                gt_file_path = os.path.join(ground_truth_dir, file)
                if os.path.exists(gt_file_path):
                    print(f"正在处理文件：{exp_file_path}")
                    # 加载对应的 Ground Truth 数据
                    with open(gt_file_path, 'r', encoding='utf-8') as f_gt:
                        ground_truth_data = [json.loads(line) for line in f_gt]
                    # 处理实验结果文件
                    process_experiment_file(exp_file_path, ground_truth_data)
                else:
                    print(f"警告：未找到对应的 Ground Truth 文件：{gt_file_path}")

if __name__ == "__main__":
    main()