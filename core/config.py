from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 전역 설정을 관리한다."""

    BACKEND_HOST: Optional[str] = None
    OLLAMA_MODEL_NAME: str
    OLLAMA_BASE_URL: str
    # MCP_SERVER_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
    )

settings = Settings()
