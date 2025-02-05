from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Mapping, Optional
from pydantic import Field

class LocalQwenLLM(LLM):
    model_name: str = Field(..., description="Name of the Qwen model")
    model: Any = Field(default=None, description="The loaded Qwen model")
    tokenizer: Any = Field(default=None, description="The loaded Qwen tokenizer")

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _llm_type(self) -> str:
        return "local_qwen"

def main():
    # 初始化本地Qwen LLM
    llm = LocalQwenLLM(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

    # 创建一个简单的提示模板
    template = "请给出'{topic}'日期下的携程上的从上海北京的高铁价格"
    prompt = PromptTemplate(template=template, input_variables=["topic"])

    # 创建LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # 运行链
    result = chain.run("2024.10.11")
    print(result)

if __name__ == "__main__":
    main()