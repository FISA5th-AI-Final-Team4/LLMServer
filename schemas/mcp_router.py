from pydantic import BaseModel
from typing import Optional
from enum import Enum


# --- 기존 스키마 ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


# --- 1단계: 쿼리 전처리 스키마 ---
class PreprocessedQuery(BaseModel):
    """전처리된 쿼리 결과"""
    original_query: str  # 원본 쿼리
    normalized_query: str  # 정제된 검색용 쿼리
    key_keywords: list[str]  # 추출된 핵심 키워드
    confidence: float  # 전처리 신뢰도 (0.0 ~ 1.0)


# --- 2단계: 쿼리 라우팅 스키마 ---
class RouteType(str, Enum):
    """라우팅 경로 타입"""
    RAG_SEARCH = "RAG_SEARCH"  # RAG 검색 (카드 추천, 금융 용어)
    ML_TOOL = "ML_TOOL"  # ML 모델 (소비 패턴 분석 등)
    QNA_DB = "QNA_DB"  # 간단한 QnA (이벤트 정보 등)
    GENERAL = "GENERAL"  # 일반 대화


class RoutingDecision(BaseModel):
    """라우팅 결정 결과"""
    route: RouteType  # 선택된 경로
    reason: str  # 선택 이유
    confidence: float  # 결정 신뢰도 (0.0 ~ 1.0)
    preprocessed_query: PreprocessedQuery  # 전처리된 쿼리 정보


# --- 통합 파이프라인 요청/응답 ---
class QueryPipelineRequest(BaseModel):
    """쿼리 파이프라인 전체 요청"""
    query: str
    skip_preprocessing: bool = False  # 전처리 생략 여부
    force_route: Optional[RouteType] = None  # 강제 라우팅 (테스트용)