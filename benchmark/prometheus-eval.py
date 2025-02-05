# Absolute Grading: Outputs score of 1 to 5

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", gpu_memory_utilization=0.9 , max_model_len=16384  )
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

# instruction = "Struggling with a recent break-up, a person opens up about the intense feelings of loneliness and sadness. They ask for advice on how to cope with the heartbreak and move forward in life.",
# response = "I'm genuinely sorry to hear about your break-up. This can be an immensely challenging time filled with intense emotions of loneliness and sorrow. It's important to remember that it's normal to experience these feelings; they are a testament to the deep connection you once had. Firstly, don't hesitate to embrace your emotions, let them flow freely. Ignoring them often complicates the healing process. It's natural to be upset, to shed tears, or to feel angry. This is all part of the journey towards healing. Self-care can be a healing balm in these tough times. This could mean indulging in your favourite book, journaling your thoughts, or even just a quiet cup of tea. Small acts of kindness to oneself can significantly improve your emotional well-being. Also, don't isolate yourself. Surround yourself with supportive individuals who can offer comfort and understanding. This might include friends, family, or even a mental health professional. They can provide much-needed emotional support and strategies to help you cope. Lastly, remember there's no set timeline for healing. It's different for everyone and it's okay to take as long as you need. Keep your focus on the future and the potential it holds. Though it may not seem possible now, you will emerge stronger from this experience. It's okay to ask for help, it's okay to grieve, and it's okay to take it one day at a time. You're not alone in this journey, and there are people ready and willing to help. It's okay to heal at your own pace.",
#reference_answer = "I can only imagine how difficult this time must be for you. Heartbreak can feel like an overwhelming wave that washes over you, making you feel lost and alone. It's important to remember that it's okay to feel this way; your feelings are valid and it's natural to grieve the loss of a relationship. Firstly, it's crucial to allow yourself to feel the emotions fully. Suppressing or denying them often only prolongs the healing process. It's okay to cry, to feel sad, or to be angry. These are all part of the healing journey. Engaging in self-care activities can also provide a much-needed boost. This could be anything from taking a warm bath, writing in a journal, or simply drinking a cup of your favorite tea. Small acts of self-love can make a big difference in how you feel. Next, try to surround yourself with supportive people who understand your situation and provide comfort. Friends and family can be a great source of strength in times of heartbreak. If you feel comfortable, you might want to consider seeking professional help. Therapists and counselors are trained to provide assistance and tools to navigate through difficult times like these. Lastly, it's important to remember that it's okay to take your time to heal. Everyone has their own pace and there's no rush. Try to focus on the future and the possibilities it holds. While it may not seem like it now, you will come out stronger and more resilient from this experience. Remember, it's okay to ask for help and it's okay to feel the way you feel. You are not alone in this journey and there are people who care about you and want to help. It's okay to take one day at a time. Healing is a process, and it's okay to move through it at your own pace.",

instruction = "针对酒店餐饮后除基础改造和安全问题我们应该如何具体实施改造计划并提升服务态度以确保总市场提出的统一解决这些问题的目标能够实现。",
response = "针对酒店餐饮后厨改造和提升服务态度，具体方案如下：1. 安全改造：升级燃气使用安全措施，更新相关设备。2. 提升服务态度：优化接待流程，明确服务标准（微笑、热情）。3. 记录并上报：将所需更换的设备清单及改造方案及时提交。4. 员工培训：定期组织服务态度培训，提升员工服务质量。通过上述措施，确保酒店尽快恢复正常运营，并提升顾客满意度。",


rubric_data = {
  "criteria":"Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",
  "score1_description":"The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",
  "score2_description":"The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",
  "score3_description":"The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",
  "score4_description":"The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",
  "score5_description":"The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation."
}

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

#reference_answer替换成ground_truth_answer
feedback, score = judge.single_absolute_grade(
    instruction=instruction,
    response=response,
    rubric=score_rubric,
    #reference_answer=reference_answer
)

print("Feedback:", feedback)
print("Score:", score)

# Output
# Feedback: The response provided shows a high level of empathy and emotional intelligence. It effectively addresses the emotional distress expressed by the user. It acknowledges the user's pain and validates their feelings of loneliness and sadness, which is a crucial aspect of providing empathetic advice. The response also suggests practical steps for coping, such as embracing emotions, practicing self-care, and seeking support from friends, family, or professionals. Furthermore, the response reassures the user that healing is a personal process with no fixed timeline, offering comfort and understanding. It emphasizes the user's worth and potential to overcome the situation, which demonstrates a profound comprehension of the user's emotions and situation. By comparing the score rubric with the provided response, it is clear that the model exhibits an excellent ability to apply empathy and emotional intelligence. The response does not have any deficiencies in emotional depth and successfully meets the criteria for a score of 5.
# Score: 5