from src.models import GeneratorProvider
from .openai_generator import OpenAIGenerator
from .lmstudio_generator import LMStudioGenerator
from src.utils import Logger


class GeneratorFactory:

    @staticmethod
    def create(generator_config):

        if generator_config.provider == GeneratorProvider.OPENAI:
            return OpenAIGenerator(
                model_name=generator_config.model_name,
                api_key=generator_config.api_key,
                base_url=generator_config.base_url,
                temperature=generator_config.temperature,
            )
        elif generator_config.provider == GeneratorProvider.LMSTUDIO:
            return LMStudioGenerator(
                model_name=generator_config.model_name,
                api_key=generator_config.api_key,
                base_url=generator_config.base_url,
                temperature=generator_config.temperature,
            )

        else:
            print(f"Unsupported generator: {generator_config.provider}")
            raise ValueError(f"Unsupported generator: {generator_config.provider}")

