from langchain_openai import ChatOpenAI

from .base import BaseGenerator


class OpenAIGenerator(BaseGenerator):

    def __init__(self, model_name, temperature, api_key):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def generate(
        self,
        question,
        context,
        system_prompt=None
    ):
        prompt = f"""
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        """

        return self.llm.invoke(prompt).content