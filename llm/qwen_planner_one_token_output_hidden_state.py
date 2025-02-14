import time
import sys
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 1. 加载模型与分词器
model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def warm_up_model(model, tokenizer):
    """预热模型，同时预热特征提取"""
    warmup_prompt = "你好,请输出三个token单位"
    warmup_inputs = tokenizer([warmup_prompt], return_tensors="pt").to(model.device)
    print("=========== Warmup: Start ===========")
    with torch.no_grad():
        # 同时预热生成和特征提取
        outputs = model(**warmup_inputs, output_hidden_states=True)
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=3,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    print("=========== Warmup: End ===========\n")

@torch.no_grad()  # 使用装饰器避免梯度计算
def classify_text_optimized(hidden_states):
    """
    优化并添加调试信息的分类函数
    """
    # 记录开始时间
    start_time = time.time()
    
    # 1. 获取最后一层特征
    t1 = time.time()
    last_layer = hidden_states[-1]
    print(f"获取最后一层耗时: {time.time() - t1:.5f} 秒")
    print(f"最后一层形状: {last_layer.shape}")
    
    # 2. 获取最后一个token特征
    t2 = time.time()
    last_token_feature = last_layer[:, -1]
    print(f"获取最后token耗时: {time.time() - t2:.5f} 秒")
    print(f"最后token特征形状: {last_token_feature.shape}")
    
    # 3. 确保数据在正确的设备上
    t3 = time.time()
    device = last_token_feature.device
    print(f"特征所在设备: {device}")
    print(f"检查设备耗时: {time.time() - t3:.5f} 秒")
    
    # 4. 计算均值
    t4 = time.time()
    with torch.no_grad():  # 确保不计算梯度
        mean_value = torch.mean(last_token_feature.float())  # 确保使用float类型
    print(f"计算均值耗时: {time.time() - t4:.5f} 秒")
    print(f"均值: {mean_value.item()}")
    
    # 5. 返回结果
    t5 = time.time()
    result = "1" if mean_value > 0 else "0"
    print(f"结果判断耗时: {time.time() - t5:.5f} 秒")
    
    total_time = time.time() - start_time
    print(f"分类函数总耗时: {total_time:.5f} 秒")
    
    return result

@torch.no_grad()  # 使用装饰器避免梯度计算
def classify_text_optimized_v2(hidden_states):
    """
    更高效的分类函数版本
    """
    with torch.no_grad():
        # 直接在一行中完成所有操作，避免中间变量
        mean_value = hidden_states[-1][:, -1].float().mean()
        return "1" if mean_value > 0 else "0"


def stream_generate_and_print(model, tokenizer, generation_kwargs):
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )
    token_times = []
    streamed_text = ""
    
    def generate_thread_func():
        model.generate(**generation_kwargs, streamer=streamer)

    gen_thread = Thread(target=generate_thread_func)
    gen_thread.start()

    prev_time = time.time()
    for new_token in streamer:
        now_time = time.time()
        token_time = now_time - prev_time
        prev_time = now_time
        token_times.append(token_time)
        sys.stdout.write(new_token)
        sys.stdout.flush()
        streamed_text += new_token

    gen_thread.join()
    return streamed_text, token_times

def main():
    # 执行预热
    warm_up_model(model, tokenizer)

    prompt = """
    ### 要求 ###
    首先你需要判断以下问题是否是简短问题：
    - 如果问题少于90字是简短问题，只输出："1"
    - 如果问题超过90字或包含"基于之前"等类似字样是长字数问题， 输出："0"
    - 如果并判断完是简短问题，输出完"1"后则不需要继续输出
    - 如果判断出是长字数问题，在输出完单个"0"后根据示例输出,你可以使用以下工具，只能选择使用信息检索RAG。
    ###

    ### 工具 ###
    [
    Tool(
    name="信息检索RAG",
    func=lambda action_input: information_retrieval_rag(action_input, planner_llm, meeting_transcript),
    description=(
    "利用实时记录的会议信息，根据问题提供关键词作为行动输入，然后根据该关键词在实时记录的会议信息"
    "遇到需要从会议记录中检索信息的问题时使用此工具。行动输入应包含关键词和问题,关键词越短越好。"
    "只在需要用到会议之前的信息的时候才选择使用此工具,针对问题时不需要使用该工具。"
    )
    )
    ]
    ###

    ### 示例简短问题 ###
    大学生就业选择大公司还是小公司各部门有哪些具体建议这个管理
    ###

    ### 示例输出1 ###
    1
    ###

    ### 示例长字数问题 ###
    基於之前我們討論的內容 關於新開業攝影樓的宣傳和營銷策略你怎麼看待我們提出的前三天打五折活動 以及定期抽選幸運顧客免費拍攝的方案再稍微算Time片這些活動是否能有效提升我們的知名度和吸引新客戶同时我们应该如何平衡优惠活动和成本控制以确保公司的长期盈利
    ###

    ### 示例输出2 ###
    0
    1. 行动：信息检索RAG
    2. 行动输入：关键词：免費拍攝 打五折活動 优惠活动 吸引新客戶
    ###

    ### 你需要判断的问题###
    基於之前我們討論的內容 關於新開業攝影樓的宣傳和營銷策略你怎麼看待我們提出的前三天打五折活動 以及定期抽選幸運顧客免費拍攝的方案再稍微算Time片這些活動是否能有效提升我們的知名度和吸引新客戶同时我们应该如何平衡优惠活动和成本控制以确保公司的长期盈利
    ###
    """
    #如何通过创新美容项目和提升服务质量来吸引新客户然后你就会觉得不舒服
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # # 计时开始
    # classification_start = time.time()
    
    # # 计时开始
    # total_start = time.time()
    
    with torch.no_grad():
        # 前向传播
        forward_start = time.time()
        outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        forward_time = time.time() - forward_start
        print(f"前向传播耗时: {forward_time:.5f} 秒")
        
        
        # 先确保 GPU 操作完成
        torch.cuda.synchronize()
        

        # 使用优化版本，并确保时间测量紧贴着函数执行
        classify_start_v2 = time.time()
        classification_result = classify_text_optimized_v2(hidden_states)
        torch.cuda.synchronize()  # 确保 GPU 操作完成后再停止计时
        classify_time_v2 = time.time() - classify_start_v2
        print(f"优化版分类函数耗时: {classify_time_v2:.5f} 秒")
        
        # total_time = time.time() - forward_start
        # print(f"总耗时: {total_time:.5f} 秒")
    
    
    # 第一次生成
    print("\n========== 第一次生成(判断 0/1) ==========")
    first_generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    with torch.no_grad():
        first_gen_text, first_token_times = stream_generate_and_print(
            model=model,
            tokenizer=tokenizer,
            generation_kwargs=first_generation_kwargs
        )

    print("\n[第一阶段] 每个 token 耗时:")
    for i, t in enumerate(first_token_times):
        print(f"  - Token {i+1}: {t:.5f} 秒")

    # 解析生成的token
    original_input_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
    index = first_gen_text.find(original_input_text)
    if index != -1:
        new_gen_part = first_gen_text[index + len(original_input_text):]
    else:
        new_gen_part = first_gen_text

    judge_str = new_gen_part.strip()
    judge_token = judge_str[-1] if len(judge_str) > 0 else ""
    print("判定 token:", judge_token)

    # 根据分类结果决定是否继续生成
    if classification_result == "1":
        print("最终输出: 1")
    else:
        print("\n========== 第二次生成(继续流式输出) ==========")
        second_generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        with torch.no_grad():
            second_gen_text, second_token_times = stream_generate_and_print(
                model=model,
                tokenizer=tokenizer,
                generation_kwargs=second_generation_kwargs
            )
            
        print("\n[第二阶段] 每个 token 耗时:")
        for i, t in enumerate(second_token_times):
            print(f"  - Token {i+1}: {t:.5f} 秒")

if __name__ == "__main__":
    main()