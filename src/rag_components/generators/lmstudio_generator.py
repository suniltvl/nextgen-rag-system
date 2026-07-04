from .base import BaseGenerator
from langchain_openai import ChatOpenAI


class LMStudioGenerator(BaseGenerator):

    def __init__(self, base_url, model_name, temperature, api_key):

        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            extra_body={
                "reasoning": {
                    "enabled": False
                }
            }
        )

    def generate(self, question, context, system_prompt=None):

        prompt = f"""
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        """

        return self.llm.invoke(prompt).content