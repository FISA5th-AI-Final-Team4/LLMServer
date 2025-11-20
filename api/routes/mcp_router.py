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
from core.parse_tool import parse_tool_from_messages

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
    1. 사용자 쿼리 입력
    2. LLM이 적절한 Tool 자동 선택
    3. MCP 서버의 Tool 실행
    4. 결과 반환
    
    지원 Tool (MCP 서버에서 제공):
    - get_card_recommendation: 카드 추천
    - analyze_consumption_pattern: 소비 패턴 분석
    - query_faq_database: FAQ 조회
    - 기타 MCP 서버에서 등록한 모든 Tool
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
    tool_response = parse_tool_from_messages(state_messages)

    return QueryResponse(
        answer=ai_text,
        tool_response=tool_response
    )