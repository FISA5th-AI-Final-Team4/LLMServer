from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 전역 설정을 관리한다."""

    BACKEND_HOST: Optional[str] = None
    OLLAMA_MODEL_NAME: str
    OLLAMA_BASE_URL: str
    MCP_SERVER_URL: str

    # --- 쿼리 전처리 및 라우팅 설정 ---
    # 전처리에 사용할 LLM 모델 (기본값: OLLAMA_MODEL_NAME과 동일)
    PREPROCESSING_MODEL_NAME: Optional[str] = None
    
    # 라우팅에 사용할 LLM 모델 (기본값: OLLAMA_MODEL_NAME과 동일)
    ROUTING_MODEL_NAME: Optional[str] = None
    
    # LLM 온도 설정
    PREPROCESSING_TEMPERATURE: float = 0.1  # 전처리는 정확성 우선 (낮은 온도)
    ROUTING_TEMPERATURE: float = 0.0  # 라우팅은 일관성 우선 (매우 낮은 온도)
    
    # 신뢰도 임계값 설정
    MIN_PREPROCESSING_CONFIDENCE: float = 0.5  # 전처리 최소 신뢰도
    MIN_ROUTING_CONFIDENCE: float = 0.6  # 라우팅 최소 신뢰도

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
    )

settings = Settings()
