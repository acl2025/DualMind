import json
import glob
import os

def analyze_jsonl_files(directory_path):
    # 存储所有得分
    
    all_truth_scores = []
    all_meet_user_needs_scores = []
    all_simplicity_scores = []
    all_structure_scores = []
    all_completeness_scores = []
    all_synthetic_scores = []
    
    # 获取目录下所有jsonl文件
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    
    for file_path in jsonl_files:
        print("="*50)
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    truth_score = data.get("事实正确性")
                    meet_user_needs_score = data.get("满足用户需求")
                    simplicity_score = data.get("简洁度")
                    structure_score = data.get("结构性")
                    completeness_score = data.get("完整性")
                    synthetic_score = data.get("综合得分")
                    
                    question = data.get("question")
                    
                    if question is not None:
                        all_truth_scores.append(truth_score)
                        all_meet_user_needs_scores.append(meet_user_needs_score)
                        all_simplicity_scores.append(simplicity_score)
                        all_structure_scores.append(structure_score)
                        all_completeness_scores.append(completeness_score)
                        all_synthetic_scores.append(synthetic_score)
                        
                        # 检查是否是特定的文件

                        print(f"问题: {question}")
                        print(f"事实正确性: {truth_score}")
                        print(f"满足用户需求: {meet_user_needs_score}")
                        print(f"简洁度: {simplicity_score}")
                        print(f"结构性: {structure_score}")
                        print(f"完整性: {completeness_score}")
                        print(f"综合得分: {synthetic_score}")
                        print("-" * 50)
                            
                except json.JSONDecodeError:
                    continue
    
    # 计算平均分
    if all_synthetic_scores:
        average_truth_score = sum(all_truth_scores) / len(all_truth_scores)
        average_meet_user_needs_score = sum(all_meet_user_needs_scores) / len(all_meet_user_needs_scores)
        average_simplicity_score = sum(all_simplicity_scores) / len(all_simplicity_scores)
        average_structure_score = sum(all_structure_scores) / len(all_structure_scores)
        average_completeness_score = sum(all_completeness_scores) / len(all_completeness_scores)
        average_synthetic_score = sum(all_synthetic_scores) / len(all_synthetic_scores)
        
        print(f"\n事实正确性平均值: {average_truth_score:.2f}")
        print(f"\n满足用户需求平均值: {average_meet_user_needs_score:.2f}")
        print(f"\n简洁度平均值: {average_simplicity_score:.2f}")
        print(f"\n结构性平均值: {average_structure_score:.2f}")
        print(f"\n完整性平均值: {average_completeness_score:.2f}")
        print(f"\n所有问题的综合得分平均值: {average_synthetic_score:.2f}")
    else:
        print("没有找到有效的得分数据")

# 运行分析
#directory_path = "/home/leon/agent/benchmark/result_compassJudger_1214_use_experiment_question"
directory_path = "/home/leon/agent/benchmark/result_compassJudger_train_S_0106_add_text_snippet"
#directory_path = "/home/leon/agent/benchmark/result_compassJudger_train_S_0104"
analyze_jsonl_files(directory_path)
