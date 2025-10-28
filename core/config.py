from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Optional


class Settings(BaseSettings):
    # 시스템 환경변수 적용
    BACKEND_HOST: Optional[str] = None

    # .env 환경변수 파일 로드
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

# 변수로 저장하여 사용
settings = Settings()