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
from core.trace_agent import pick_last_ai_text, _log_agent_trace


router = APIRouter(tags=["MCP Client Dispatch"], prefix="/mcp-router")

@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""
    return payload

# =====================================================
# 메인 디스패치 엔드포인트 (query_router_agent 사용)
# =====================================================
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
    _log_agent_trace(state_messages)
    ai_text = pick_last_ai_text(state_messages) or ""

    return QueryResponse(answer=ai_text)