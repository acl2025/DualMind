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
    output_jsonl_path = f"/home/leon/agent/baseline/result_compassJudger_Qwen2.5-7B-Instruct_train_S_0113_add_text_snippet/result_compassJudger_{file_name}.jsonl"
    output_txt_path   = f"/home/leon/agent/baseline/result_compassJudger_Qwen2.5-7B-Instruct_train_S_0113_add_text_snippet/result_compassJudger_{file_name}.txt"

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
            question_type = 'simple' if question_len <= 60 else 'complex'
            
            if question_type == 'complex':
                text_snippet = exp_result.get('context', '').strip()
            else:
                text_snippet = ""
                
                
            # 使用正则表达式截取text_snippet中从开头到###End RAG used context:###的部分
            if text_snippet:
                match = re.search(r'^.*?###End RAG used context:###', text_snippet, re.DOTALL)
                if match:
                    text_snippet = match.group(0)

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
            score_keys = ["事实正确性", "满足用户需求", "简洁度", "结构性", "完整性" , "综合得分"]
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
    if 'answer' in exp_result:
        assistant_answer = exp_result['answer']

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
        prompt_template = """你是一个擅长会议agent助手回复质量的助手。
    请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是角色扮演，因此你需要从下面的几个维度对回答进行评估:
    1. 事实正确性: 回答中提供的信息是否准确无误，是否基于可信的事实和数据。
    2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
    3. 简洁度: 回答是否简洁明了，是否避免了冗余和重复，字数少很重要,字数少是加分项。
    4. 结构性: 回答的组织是否清晰,重点是否突出,便于用户快速理解。
    5. 完整性: 回答是否大部分覆盖了问题相关的会议内容,是否有重要信息遗漏。


    我们会给您提供用户的提问，可能是高质量的参考答案，和需要你评估的AI助手的答案,如果AI助手的答案是空的，请给0分。如果参考答案开头说会议内容中没有提及相关内容类似的话，但是助手的答案中却提到了相关内容，那么说明助手的答案更好、检索到了更多会议内容，应给10分满分。

    当你开始你的评估时，你需要遵守以下流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释,请注意，参考答案并不一定比AI助手的答案更好，参考答案评分水准在3-10分不等，请据此判断参考答案的水平。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给出1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：
    - 事实错误或无关/有害等，要给低分(1~2)；
    - 没有严重错误，但质量较低，也给低分(3~4)；
    - 基本满足要求但在部分维度较差，可给中等分(5~7)；
    - 与参考答案相近或略差，可给8~9分；
    - 若超越参考答案，更简短，且各维度都近乎满分，可给10分。
    作为示例，参考答案可以得10分。

    最后，请在回答的末尾，以字典格式（包括大括号）给出您的打分结果，键分别是：
    {{'事实正确性': X, '满足用户需求': X, '简洁度': X, '结构性': X, '完整性': X, '综合得分': X}}。
    请记得在打分前先进行评估和解释，并保证每个分数是 1～10 的整数。

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
        prompt_template = """你是一个擅长会议agent助手回复质量的助手。
        请你以公正的评判者的身份，基于参考会议内容评估一个AI助手对于用户复杂提问的回答的质量。由于您评估的回答类型是角色扮演，因此你需要从下面的几个维度对回答进行评估:
    1. 事实正确性: 回答中提供的信息是否准确无误，是否基于可信的事实和数据。
    2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
    3. 简洁度: 回答是否简洁明了，是否避免了冗余和重复，字数少很重要,字数少是加分项。
    4. 结构性: 回答的组织是否清晰,重点是否突出,便于用户快速理解。
    5. 完整性: 回答是否大部分覆盖了问题相关的会议内容,是否有重要信息遗漏。


    我们会给您提供用户的提问，可能是高质量的参考答案，和需要你评估的AI助手的答案，如果AI助手的答案是空的，请给0分。如果参考答案开头说会议内容中没有提及相关内容类似的话，但是助手的答案中却提到了相关内容，那么说明助手的答案更好、检索到了更多会议内容，应给10分满分。
    
    [参考会议内容开始]
    {text_snippet}
    [参考会议内容结束]

    当你开始你的评估时，你需要遵守以下流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释,请注意，参考答案并不一定比AI助手的答案更好，参考答案评分水准在3-10分不等，请据此判断参考答案的水平。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给出1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：
    - 事实错误或无关/有害等，要给低分(1~2)；
    - 没有严重错误，但质量较低，也给低分(3~4)；
    - 基本满足要求但在部分维度较差，可给中等分(5~7)；
    - 与参考答案相近或略差，可给8~9分；
    - 若超越参考答案，更简短，且各维度都近乎满分，可给10分。
    - 如果问题语种和助手的答案语种不一致，那么助手的答案得分要减少。
    作为示例，参考答案可以得10分。

    最后，请在回答的末尾，以字典格式（包括大括号）给出您的打分结果，键分别是：
    {{'事实正确性': X, '满足用户需求': X, '简洁度': X, '结构性': X, '完整性': X, '综合得分': X}}。
    请记得在打分前先进行评估和解释，并保证每个分数是 1～10 的整数。
    
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
            text_snippet=text_snippet
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
    #exp_results_dir = "/home/leon/agent/baseline/result_Qwen2Audio-7B-Instruct_baseline"
    exp_results_dir = "/home/leon/agent/baseline/result_Qwen2.5-7B-Instruct_baseline"
    
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
