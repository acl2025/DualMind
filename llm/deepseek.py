from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re  # 新增正则表达式库
from torch.cuda.amp import autocast

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型并移动到GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,    
    torch_dtype="auto",
    device_map="auto"
)

# 定义输入文本
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

# 修改系统提示，明确禁止输出<think>
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. 请直接输出最终答案，不要包含任何思考过程（如<think>标签）。"
    },
    {"role": "user", "content": prompt}
]

# 使用tokenizer处理输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 将输入数据移动到GPU
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 使用混合精度训练生成文本
with autocast():  # 启用混合精度
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,  # 关闭随机采样
        temperature=0.0,  # 完全确定性输出
        eos_token_id=tokenizer.eos_token_id
    )

# 解码生成的文本并过滤<think>标签
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()  # 过滤标签

print(response)