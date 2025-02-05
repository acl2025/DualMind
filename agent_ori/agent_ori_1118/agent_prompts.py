from langchain.prompts import PromptTemplate
from agent.agent_utils import CustomPromptTemplate
from agent.agent_tools import tools

# talker_prompt = PromptTemplate(
#     template="""### 问题：###
#     {input}
#     ###
    
#     ### 要求 ###
#     如果不需要之前的会议信息，请只用100字以内回答之前问题。 如果需要用之前的会议信息,请只回复：“请使用慢思考Reasoner”。
#     ###
#     """,
#     input_variables=["input"]
# )

talker_prompt = PromptTemplate(
    template="""
    {input},只用100字以内回答
    """,
    input_variables=["input"]
)

classifier_prompt = PromptTemplate(
    template="""
    ### 要求 ###
    判断以下你需要判断的问题是否为可直接回答的问题，如果该问题是可以直接回答的针对性问题，只输出"1"。 如果该问题需要你告知之前的会议信息内容,只输出"0"。注意绝对不要输出额外解释文字，只输出0或1.
    ###
    
    ### 示例针对性问题1 ###
    针对不同年龄段顾客的购物习惯，我们如何在周一至周五针对老年人推广高钙奶和无糖产品，而在周末针对儿童和青年人推广奶制品，并确保我们的送货服务标准既能满足顾客需求又不会导致过高的人工成本？
    ###
    
    ### 示例输出1 ###
    1
    ###
    
    ### 示例会议相关问题2 ###
    在讨论学生上课注意力不集中的问题时，有提到老师会通过课堂点名来提醒，那么具体有哪些方法被用来提高学生的课堂参与度和专注力呢？
    ###
    
    ### 示例输出2 ###
    0
    ###
    
    ### 你需要判断的问题（你绝对只输出0或1） ###
    {input}
    ###
    """,
    input_variables=["input"]
)

# 创建 Reasoner 的 PromptTemplate 和 LLMChain，与之前相同
reasoner_prompt = CustomPromptTemplate(
    template="""请回答以下问题。你可以使用以下工具：
### 工具 ###
{tools}
###
如果判断出不需要工具，便直接回答问题,回复100字以内,请参照以下输入的问题示例和示例输出。

你可以从以下格式和回答元素中进行选择：

### 格式 ###
1. 行动：要采取的行动，可以是 [{tool_names}] 中的一个，或者你可以直接自己回答
2. 行动输入：行动的输入
###

### 输入的问题示例1 ###
会议助手agent：在讨论交通管理问题时，有提到需要增加人员进行巡视以避免危险发生，那么具体计划增加多少人手来加强这方面的工作呢？
### 

### 示例输出1 ###
1. 行动：信息检索RAG
2. 行动输入：关键词：交通管理 增加人员
###

### 输入的问题示例2 ###
100字以内介绍一下智能机器人儿的应用 。
### 

### 示例输出2 ###
智能机器人的应用非常广泛，可以在客服、医疗、教育、交通等多个领域提供自动化服务，提高效率和准确性。例如，在客服领域，智能机器人可以解答常见问题，减轻人工负担；在医疗领域，智能机器人能辅助诊断和监测患者状况。
###

问题：{input}
{agent_scratchpad}""",

    tools=tools,
    input_variables=["input", "intermediate_steps"]
)
