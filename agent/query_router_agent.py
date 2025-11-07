"""
쿼리 라우터 에이전트
- 1단계: 쿼리 전처리 (Query Preprocessing)
- 2단계: 쿼리 라우팅 (Query Routing)
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

import json
from typing import Dict, Any

from core.config import settings
from schemas.mcp_router import PreprocessedQuery, RouteType, RoutingDecision


# --- LLM 인스턴스 생성 ---
# 전처리용 LLM (정확성 우선)
preprocessing_llm = ChatOllama(
    model=settings.PREPROCESSING_MODEL_NAME or settings.OLLAMA_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=settings.PREPROCESSING_TEMPERATURE,
    request_timeout=60.0
)

# 라우팅용 LLM (일관성 우선)
routing_llm = ChatOllama(
    model=settings.ROUTING_MODEL_NAME or settings.OLLAMA_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=settings.ROUTING_TEMPERATURE,
    request_timeout=60.0
)


# =====================================================
# 1단계: 쿼리 전처리 (Query Preprocessing)
# =====================================================

def preprocess_query(query: str) -> PreprocessedQuery:
    """
    사용자의 원본 쿼리를 벡터 검색에 최적화된 형태로 전처리합니다.
    
    수행 작업:
    - 구어체 → 검색용 표준 표현으로 변환
    - 핵심 키워드 추출
    - 오타 수정 및 약어 해소
    - 불필요한 조사/어미 제거
    
    Args:
        query: 사용자의 원본 쿼리
        
    Returns:
        PreprocessedQuery: 전처리 결과 (정제된 쿼리, 키워드, 신뢰도)
    """
    
    print(f"\n{'='*60}")
    print(f"[1단계: 쿼리 전처리 시작]")
    print(f"원본 쿼리: {query}")
    print(f"{'='*60}")
    
    # 시스템 프롬프트: 전처리 지침
    system_prompt = """당신은 사용자 쿼리를 벡터 검색에 최적화된 형태로 전처리하는 전문가입니다.

**당신의 임무**:
1. 사용자의 구어체 쿼리를 명확하고 검색 가능한 표준 표현으로 변환
2. 핵심 키워드를 3~5개 추출 (카드 혜택, 소비 패턴 관련)
3. 오타 수정, 약어 해소, 불필요한 조사 제거
4. 전처리 결과의 품질을 0.0~1.0으로 평가

**예시**:
입력: "20대 여잔데 쇼핑 자주함"
출력: 
{
  "normalized_query": "20대 여성 쇼핑 할인 카드 추천",
  "key_keywords": ["20대", "여성", "쇼핑", "할인", "카드"],
  "confidence": 0.85
}

입력: "편의점 많이씀 할인되는거"
출력:
{
  "normalized_query": "편의점 할인 카드 추천",
  "key_keywords": ["편의점", "할인", "카드", "추천"],
  "confidence": 0.9
}

**중요 규칙**:
- normalized_query는 반드시 한국어로 작성
- key_keywords는 정확히 3~5개
- confidence는 전처리 품질에 대한 자신감 (높을수록 좋음)
- 응답은 반드시 JSON 형식만 출력 (다른 텍스트 금지)"""

    user_prompt = f"다음 쿼리를 전처리하세요:\n\n{query}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        # LLM 호출
        print(f"[전처리 LLM 호출 중...]")
        response = preprocessing_llm.invoke(messages)
        raw_output = response.content
        
        print(f"[LLM 원본 응답]\n{raw_output}\n")
        
        # JSON 파싱
        # LLM이 ```json ... ``` 형태로 응답할 수 있으므로 처리
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            json_str = raw_output.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_output.strip()
        
        parsed_data = json.loads(json_str)
        
        # PreprocessedQuery 객체 생성
        result = PreprocessedQuery(
            original_query=query,
            normalized_query=parsed_data.get("normalized_query", query),
            key_keywords=parsed_data.get("key_keywords", []),
            confidence=float(parsed_data.get("confidence", 0.5))
        )
        
        print(f"[전처리 완료]")
        print(f"✓ 정제된 쿼리: {result.normalized_query}")
        print(f"✓ 핵심 키워드: {', '.join(result.key_keywords)}")
        print(f"✓ 신뢰도: {result.confidence:.2f}")
        print(f"{'='*60}\n")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 실패: {e}")
        print(f"[폴백] 원본 쿼리 사용")
        
        # 파싱 실패 시 원본 쿼리를 그대로 사용 (신뢰도 낮게)
        return PreprocessedQuery(
            original_query=query,
            normalized_query=query,
            key_keywords=[],
            confidence=0.3
        )
        
    except Exception as e:
        print(f"[오류] 전처리 실패: {e}")
        print(f"[폴백] 원본 쿼리 사용")
        
        return PreprocessedQuery(
            original_query=query,
            normalized_query=query,
            key_keywords=[],
            confidence=0.0
        )


# =====================================================
# 2단계: 쿼리 라우팅 (Query Routing) - 다음 단계에서 구현
# =====================================================

def route_query(preprocessed_query: PreprocessedQuery) -> RoutingDecision:
    """
    전처리된 쿼리를 분석하여 적절한 도구로 라우팅합니다.
    
    TODO: 다음 단계에서 구현
    """
    pass


# =====================================================
# 통합 파이프라인 (다음 단계에서 구현)
# =====================================================

class QueryRouterAgent:
    """
    쿼리 전처리 + 라우팅을 통합한 에이전트
    
    TODO: 다음 단계에서 구현
    """
    pass
