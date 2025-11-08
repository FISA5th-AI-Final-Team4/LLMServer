from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from schemas.mcp_router import QueryRequest, QueryResponse
from agent.weather_agent import weather_agent
from agent.query_router_agent import query_router_agent

router = APIRouter(tags=["MCP Client Dispatch"], prefix="/mcp-router")


@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""
    return payload


# =====================================================
# 날씨 데모 엔드포인트 (weather_agent 사용)
# =====================================================

@router.post("/dispatch-demo/weather", response_model=QueryResponse)
def chat(req: QueryRequest):
    """
    [날씨 데모] 날씨 Tool 호출 테스트
    
    weather_agent를 사용하여 날씨 조회를 테스트합니다.
    """
    system_hint = (
        "당신은 도우미입니다. 사용자가 특정 도시의 현재 날씨/기온을 묻는다면 "
        "'get_weather' 도구를 반드시 호출해 답을 구성하세요. "
        "그 외 일반 질문은 도구 없이 직접 답하세요. "
        "도구 인자는 city (예: 'Seoul', 'New York')."
    )
    user_q = f"{system_hint}\n\n질문: {req.query}"
    
    result = weather_agent.run(user_q)
    return QueryResponse(answer=result)


@router.post("/stream-dispatch")
async def stream_dispatch(req: QueryRequest):
    """
    [스트리밍 데모] 비동기 스트리밍 응답
    
    weather_agent를 사용하여 스트리밍 응답을 테스트합니다.
    JSON Line 형식: {type: ..., payload: ...}
    """
    print(f"LLM (Stream): 프롬프트 수신 -> '{req.query}'")
    
    return StreamingResponse(
        weather_agent.arun(req.query),
        media_type="application/x-json-stream"
    )


# =====================================================
# 메인 디스패치 엔드포인트 (query_router_agent 사용)
# =====================================================

@router.post("/dispatch", response_model=QueryResponse)
def dispatch(req: QueryRequest):
    """
    [메인 디스패치] 쿼리 라우터 기반 자동 Tool 선택
    
    사용자 쿼리를 받아 자동으로 적절한 Tool을 선택하고 실행합니다.
    - LangChain Tool 기반 자동 라우팅
    - 쿼리 전처리 (구어체 → 검색 최적화)
    - MCP 서버 연동 (operation_id 자동 매칭)
    
    지원 Tool:
    - get_card_recommendation: 카드 추천
    - analyze_consumption_pattern: 소비 패턴 분석
    - query_faq_database: FAQ 조회
    - 직접 답변: 일반 대화
    
    예시:
    - "편의점 할인 카드 추천해줘" → 카드 추천 Tool
    - "내 소비 패턴 분석해줘" → 소비 패턴 Tool
    - "이벤트 언제까지?" → FAQ Tool
    - "안녕하세요" → 직접 답변
    """
    print(f"\n[/dispatch] 요청: {req.query}")
    
    result = query_router_agent.run(req.query)
    
    return QueryResponse(answer=result)


# =====================================================
# 테스트 엔드포인트
# =====================================================

@router.post("/test-preprocessing", response_model=dict)
def test_preprocessing(req: QueryRequest):
    """
    [테스트 전용] 쿼리 전처리 결과 확인
    
    1단계(쿼리 전처리)의 결과를 확인합니다.
    """
    from agent.query_router_agent import _preprocess_query_internal
    
    print(f"\n[전처리 테스트] 요청: {req.query}")
    
    preprocessed = _preprocess_query_internal(req.query)
    
    return {
        "original_query": preprocessed.original_query,
        "normalized_query": preprocessed.normalized_query,
        "key_keywords": preprocessed.key_keywords,
        "confidence": preprocessed.confidence
    }