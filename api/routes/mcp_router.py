from typing import List, Optional, Any

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)

from core.dep import AgentDep
from core.model_config import SYSTEM_PROMPT
from schemas.mcp_router import QueryRequest, QueryResponse
# from agent.weather_agent import weather_agent
# from agent.query_router_agent import query_router_agent

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

def pick_last_ai_text(messages: List[BaseMessage]) -> Optional[str]:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            return getattr(m, "content", "")
    return None

@router.post("/dispatch", response_model=QueryResponse)
async def dispatch(req: QueryRequest, agent: AgentDep):
    """
    [메인 디스패치] 쿼리 라우터 기반 자동 Tool 선택
    
    실행 흐름:
    1. 사용자 원본 쿼리 입력 (구어체 그대로)
    2. LLM이 적절한 Tool 자동 선택 (원본 쿼리로 의도 파악)
    3. 선택된 Tool 내부에서 쿼리 전처리 (검색 최적화)
    4. 전처리된 쿼리로 MCP 서버 호출 (operation_id 자동 매칭)
    5. 결과 반환
    
    지원 Tool:
    - get_card_recommendation: 카드 추천 (RAG 검색)
    - analyze_consumption_pattern: 소비 패턴 분석 (준비 중)
    - query_faq_database: FAQ 조회 (준비 중)
    - 직접 답변: 일반 대화 (Tool 선택 안 함)
    
    예시:
    입력: "편의점 많이 쓰는데 할인 카드 추천해줘"
    → Tool 선택: get_card_recommendation (구어체로 의도 파악)
    → 내부 전처리: "편의점 할인 카드 추천" (검색 최적화)
    → MCP 호출 & 응답
    
    입력: "내 소비 패턴 분석해줘" → analyze_consumption_pattern Tool
    입력: "이벤트 언제까지?" → query_faq_database Tool
    입력: "안녕하세요" → LLM 직접 답변
    """
    print(f"\n[/dispatch] 요청: {req.query}")

    # history = get_history()
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=req.query)
    ]
        # + history # 향후 대화형 히스토리 지원 시 활성화

    # create_react_agent는 {"messages": [...]} 입력을 받습니다.
    result = await agent.ainvoke({"messages": messages})
    state_messages: List[BaseMessage] = result.get("messages", [])
    ai_text = pick_last_ai_text(state_messages) or ""

    return QueryResponse(answer=ai_text)


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