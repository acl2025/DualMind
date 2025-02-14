import json
import glob
import os

def analyze_jsonl_files(directory_path):
    # 存储所有得分
    all_scores = []
    # 存储所有问题和得分的映射
    question_scores = {}
    
    # 获取目录下所有jsonl文件
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    
    for file_path in jsonl_files:
        print("="*50)
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    score = data.get("score")
                    question = data.get("question")
                    
                    if score is not None and question is not None:
                        all_scores.append(score)
                        question_scores[question] = score
                        
                        # 检查是否是特定的文件

                        print(f"问题: {question}")
                        print(f"score: {score}")
                        print("-" * 50)
                            
                except json.JSONDecodeError:
                    continue
    
    # 计算平均分
    if all_scores:
        average_score = sum(all_scores) / len(all_scores)
        print(f"\n所有问题的score平均值: {average_score:.2f}")
    else:
        print("没有找到有效的得分数据")

# 运行分析
#directory_path = "/home/leon/agent/benchmark/result_compassJudger_1214_use_experiment_question"
#directory_path = "/home/leon/agent/benchmark/result_prometheus_eval_train_S_0104"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_Llama-2-7b-chat-hf_baseline_0115"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_Llama-2-13b-chat-hf_baseline_0115"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-7B-Instruct_baseline_0115"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2.5-14B-Instruct_baseline_0115"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_Qwen2Audio-7B-Instruct_baseline_0115"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_chatGLM3_6B_baseline_0207"
#directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_deepseek_r1_7B_baseline_0211"
directory_path = "/home/leon/agent/baseline/result_prometheus_eval_train_S_deepseek_r1_14B_baseline_0211"
analyze_jsonl_files(directory_path)
