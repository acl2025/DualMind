import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------
#  加载CompassJudger模型
# ---------------------------------
def load_model():
    model_name = "opencompass/CompassJudger-1-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# ---------------------------------
#  清洗助手回答中的无关内容
# ---------------------------------
def clean_assistant_answer(answer: str) -> str:
    """
    使用正则删除诸如 system\n...assistant\n 等内容，并去除首尾空格
    """
    # 删除类似: system\nYou are a helpful assistant.\nuser\n...assistant\n 的内容
    answer = re.sub(r"system\s*\n.*?\nuser\s*\n.*?\nassistant\s*\n", "", answer, flags=re.DOTALL)
    # 也可以根据其他可能出现的模式灵活增减清洗规则
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
    4. 调用 CompassJudger 评分
    5. 实时写入 .jsonl 和 .txt 文件
    6. 每处理一个问题后，重置模型
    """
    with open(exp_file_path, 'r', encoding='utf-8') as f:
        exp_results = [json.loads(line) for line in f]
    print(f"正在处理实验结果文件: {exp_file_path}")

    # 获取文件名
    file_name = os.path.basename(exp_file_path).replace('.jsonl', '')
    output_jsonl_path = f"/home/leon/agent/benchmark/result_compassJudger_train_S_0124_ori_judgeprompt/result_compassJudger_{file_name}.jsonl"
    output_txt_path   = f"/home/leon/agent/benchmark/result_compassJudger_train_S_0124_ori_judgeprompt/result_compassJudger_{file_name}.txt"

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
    model, tokenizer = load_model()

    # 保证输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out_jsonl, \
         open(output_txt_path, 'a', encoding='utf-8') as f_out_txt:

        for exp_result in unprocessed_questions:
            question = exp_result.get('question', '').strip()
            question_len = len(question)
            question_type = 'simple' if question_len <= 40 else 'complex'
            
            if exp_result.get('planner_first_judgetoken', '').strip() == '0':
                text_snippet = exp_result.get('text_snippet', '').strip()
            else:
                text_snippet = ""

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
                # 匹配不到就跳过
                continue

            # 取到 ground truth
            gt_question = matched_gt.get('question', '').strip()
            gt_answer   = matched_gt.get('answer', '').strip()

            # 构建 prompt（示例中将提问替换为 ground truth 的 question，您也可改回 question）
            prompt = build_compass_prompt(gt_question, gt_answer, assistant_answer,question_type,text_snippet)
            print(f"prompt: {prompt}")

            # 推理、获取打分
            response, scores = run_compass_judger(prompt, model, tokenizer)

            # 填充缺省字段
            score_keys = ["事实正确性", "满足用户需求", "逻辑连贯性", "创造性", "丰富度" , "综合得分"]
            for k in score_keys:
                if k not in scores:
                    scores[k] = 0

            # 写入 jsonl
            output_record = {
                'question': question,
                'prompt': prompt,
                'compassJudger_output': response
            }
            
            print(f"compassJudger_output: {response}")
            print(f"compassJudger_scores: {scores}")
                        
            output_record.update(scores)
            f_out_jsonl.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            f_out_jsonl.flush()

            # 写入 txt
            f_out_txt.write(f"Question: {question}\n")
            f_out_txt.write(f"Prompt:\n{prompt}\n\n")
            f_out_txt.write(f"CompassJudger Output:\n{response}\n\n")
            f_out_txt.write("-" * 50 + "\n")
            f_out_txt.flush()

            processed_questions.add(question)

            # 每处理一个问题都清理显存 + 重新加载模型，避免记忆影响
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model()

    print(f"文件 {exp_file_path} 处理完毕。")
    return

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
    if 'talker_output' in exp_result and exp_result['talker_output']:
        assistant_answer = exp_result['talker_output']
    elif 'agent_tool_result' in exp_result and exp_result['agent_tool_result']:
        assistant_answer = exp_result['agent_tool_result']

    # 清洗无关内容
    assistant_answer = clean_assistant_answer(assistant_answer)
    return assistant_answer

# ---------------------------------
#  构建 CompassJudger 使用的 prompt
# ---------------------------------
def build_compass_prompt(user_question, ground_truth_answer, assistant_answer,question_type,text_snippet):
    """
    在 prompt 中引导 CompassJudger 对回答做 6+1 个维度的打分
    """
    if question_type == 'simple':
        prompt_template = """你是一个擅长评价文本质量的助手。
        请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是角色扮演，因此你需要从下面的几个维度对回答进行评估:
            1. 事实正确性: 回答中提供的信息是否准确无误，是否基于可信的事实和数据。
            2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
            3. 逻辑连贯性: 回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾。
            4. 创造性: 回答是否具有创新性或独特性，是否提供了新颖的见解或解决方法。
            5. 丰富度: 回答包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解。
        我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
            1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
            2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
            3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
            4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。
        当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
        当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
        当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
        当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
        只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
        作为示例，参考答案可以得到8分。
        请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：
        {{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}}，例如：{{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}}。

    
        用户的提问：{user_question}
        [参考答案开始]
        {ground_truth_answer}
        [参考答案结束]
        [助手的答案开始]
        {assistant_answer}
        [助手的答案结束]"""
        prompt = prompt_template.format(
            user_question=user_question,
            ground_truth_answer=ground_truth_answer,
            assistant_answer=assistant_answer
        )
    if question_type == 'complex':
        prompt_template = """你是一个擅长评价文本质量的助手。
        请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是角色扮演，因此你需要从下面的几个维度对回答进行评估:
            1. 事实正确性: 回答中提供的信息是否准确无误，是否基于可信的事实和数据。
            2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
            3. 逻辑连贯性: 回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾。
            4. 创造性: 回答是否具有创新性或独特性，是否提供了新颖的见解或解决方法。
            5. 丰富度: 回答包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解。
        我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
            1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
            2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
            3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
            4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。
        当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
        当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
        当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
        当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
        只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
        作为示例，参考答案可以得到8分。
        请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：
        {{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}}，例如：{{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}}。

    
        用户的提问：{user_question}
        [参考答案开始]
        {ground_truth_answer}
        [参考答案结束]
        [助手的答案开始]
        {assistant_answer}
        [助手的答案结束]"""


        prompt = prompt_template.format(
            user_question=user_question,
            ground_truth_answer=ground_truth_answer,
            assistant_answer=assistant_answer,
            #text_snippet=text_snippet
        )
        
    return prompt

# ---------------------------------
#  执行 CompassJudger 推理
# ---------------------------------
def run_compass_judger(prompt, model, tokenizer):
    """
    传入 Prompt，获取 CompassJudger 的输出结果，并用 extract_scores() 函数解析评分。
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )

    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    scores = extract_scores(response)
    return response, scores

# ---------------------------------
#  从 CompassJudger 的回答中提取评分字典
# ---------------------------------
def extract_scores(response: str) -> dict:
    """
    使用正则匹配形如 { ... } 的 JSON 格式字典，并转换为 Python dict。
    """
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        scores_str = match.group()
        # 将单引号替换为双引号，以便 json.loads 能处理
        scores_str = scores_str.replace("'", '"')
        try:
            scores = json.loads(scores_str)
            return scores
        except json.JSONDecodeError:
            return {}
    return {}

# ---------------------------------
#  脚本入口
# ---------------------------------
def main():
    # Ground Truth 和实验结果目录
    ground_truth_dir = "/home/leon/agent/AISHELL_dataset/train_S/ground_truth_train_S"
    exp_results_dir = "/home/leon/agent/experiment_result/result_train_S_jsonl_audio_segment_only_0103"

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
