from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class LoggingConfig(BaseSettings):
    log_level: str = "DEBUG"

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        if isinstance(value, str):
            return value.split("#")[0].strip().upper()
        return value

    enable_console_logger: bool = True
    enable_file_logger: bool = True
    enable_json_logger: bool = True

    log_dir: str = "logs"
    log_file: str = "rag.log"

    log_retention_days: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class Settings(BaseSettings):

    logging: LoggingConfig = LoggingConfig()

    pipeline_config_file: str = os.environ.get("PIPELINE_CONFIG_FILE", "baseline.yaml")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()