from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    log_level: str = "DEBUG"

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

    pipeline_config_file: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()