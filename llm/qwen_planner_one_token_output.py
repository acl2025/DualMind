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
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

############################################################################
#               预 热 (warm-up) 步 骤
############################################################################

def warm_up_model(model, tokenizer):
    """
    给模型一个非常简短的 prompt，让它先生成一个 token。
    这样就能把 KV Cache、CUDA kernel 等初始化，减少正式推理时的首 token 延迟。
    """
    # 这里只需一小段 Prompt
    warmup_prompt = "你好,请输出三个token单位"

    # 准备输入
    warmup_inputs = tokenizer([warmup_prompt], return_tensors="pt").to(model.device)

    print("=========== Warmup: Start ===========")
    # 我们可以只生成1个token即可达到预热的目的
    with torch.no_grad():
        # 这里用非流式的 generate 就可以，主要是为了快速初始化
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=3,
            do_sample=False
        )
    print("=========== Warmup: End ===========\n")

# 执行预热
warm_up_model(model, tokenizer)


############################################################################
#               以下为正式推理逻辑，与之前的分两段示例代码相似
############################################################################

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
    如何通过创新美容项目和提升服务质量来吸引新客户然后你就会觉得不舒服
    ###
    """

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 使用 Qwen 提供的 apply_chat_template 来构造实际的聊天文本
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 转为模型的输入格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 封装一个通用的流式生成函数，用于在终端打印，并返回完整的文本内容
# 同时记录每个 token 的生成耗时列表
def stream_generate_and_print(model, tokenizer, generation_kwargs):
    """
    用 TextIteratorStreamer 实现流式输出，并打印每个 token 的生成时间。
    返回：
      - streamed_text: 本次流式生成的完整文本
      - token_times:   每个 token 的生成耗时列表
    """
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_special_tokens=True,  # 跳过特殊 token
        skip_prompt=True           # 不再重复打印 prompt
    )
    token_times = []
    streamed_text = ""

    # 模型生成时跑在线程里，我们主线程里 while True 读取 streamer 产物
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

        # 终端打印该 token（不换行）
        sys.stdout.write(new_token)
        sys.stdout.flush()

        streamed_text += new_token

    gen_thread.join()
    return streamed_text, token_times

# ============ 第一次: 只生成 1-2 个新 token，用于判断  ============
print("========== 第一次生成(判断 0/1) ==========")
first_generation_kwargs = dict(
    **model_inputs,
    max_new_tokens=1,
    do_sample=False,
    top_p=0.9,
    temperature=0.7
)
inference_start = time.time()
with torch.no_grad():
    first_gen_text, first_token_times = stream_generate_and_print(
        model=model,
        tokenizer=tokenizer,
        generation_kwargs=first_generation_kwargs
    )
inference_end = time.time()
print(f"First inference total time: {inference_end - inference_start:.4f} seconds")

print("\n[第一阶段] 每个 token 耗时:")
for i, t in enumerate(first_token_times):
    print(f"  - Token {i+1}: {t:.5f} 秒")

# 做一次差分拿到第一次生成的新 token
original_input_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
print("原始输入文本:", original_input_text)
index = first_gen_text.find(original_input_text)
if index != -1:
    new_gen_part = first_gen_text[index + len(original_input_text):]
else:
    new_gen_part = first_gen_text

judge_str = new_gen_part.strip()
judge_token = judge_str[-1] if len(judge_str) > 0 else ""

print("\n========== 第一次生成的完整文本(含prompt) ==========")
print(first_gen_text)
print("========== 第一次生成-截取得到的新生成部分 ==========")
print(new_gen_part)
print("判定 token:", judge_token)

# ============ 分支处理 ============
if judge_token == "1":
    print("最终输出: 1")

elif judge_token == "0":
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

    # 差分，拿到后续真正生成的部分
    first_full_text = first_gen_text
    index2 = second_gen_text.find(first_full_text)
    if index2 != -1:
        final_new_part = second_gen_text[index2 + len(first_full_text):]
    else:
        final_new_part = second_gen_text

    final_output = final_new_part.strip()
    print("\n========== 第二次生成的完整文本(含第一次文本) ==========")
    print(second_gen_text)
    print("========== 第二次生成-截取得到的后续生成部分 ==========")
    print(final_output)

    # 最终示例输出
    print("\n========== 最终输出(示例) ==========")
    print("0")
    print(final_output)

else:
    print("未能正确解析到 0/1，原始判定输出：", new_gen_part)
