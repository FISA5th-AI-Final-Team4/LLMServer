from pydantic import BaseModel
from typing import Optional
from enum import Enum


# --- 기본 요청/응답 스키마 ---
class QueryRequest(BaseModel):
    """쿼리 요청"""
    query: str

class QueryResponse(BaseModel):
    """쿼리 응답"""
    answer: str


# --- 쿼리 전처리 스키마 ---
class PreprocessedQuery(BaseModel):
    """전처리된 쿼리 결과"""
    original_query: str  # 원본 쿼리
    normalized_query: str  # 정제된 검색용 쿼리
    key_keywords: list[str]  # 추출된 핵심 키워드
    confidence: float  # 전처리 신뢰도 (0.0 ~ 1.0)


# --- 라우팅 관련 스키마 (Rule-based 라우팅용) ---
class RouteType(str, Enum):
    """라우팅 경로 타입"""
    RAG_SEARCH = "rag_search"  # 벡터 DB 검색 (카드 추천)
    ML_TOOL = "ml_tool"  # ML 분석 도구 (소비 패턴 분석)
    QNA_DB = "qna_db"  # FAQ 데이터베이스
    GENERAL = "general"  # 일반 대화


class RoutingDecision(BaseModel):
    """라우팅 결정 결과 (Rule-based 또는 LLM 기반)"""
    route_type: RouteType  # 선택된 경로
    reasoning: str  # 라우팅 이유
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    metadata: Optional[dict] = None  # 추가 메타데이터


class QueryPipelineRequest(BaseModel):
    """전체 쿼리 파이프라인 요청 (전처리 + 라우팅)"""
    query: str
    use_rule_based: bool = True  # True: 룰베이스, False: LLM 기반
    preprocessing_enabled: bool = True  # 전처리 활성화 여부
    confidence: float  # 전처리 신뢰도 (0.0 ~ 1.0)