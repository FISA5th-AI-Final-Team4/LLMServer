from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException, Body

from langchain_core.runnables import RunnableConfig
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
from core.history_store import get_history, persist_history


router = APIRouter(tags=["MCP Client Dispatch"], prefix="/mcp-router")

@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""
    return payload

# =====================================================
# 메인 디스패치 엔드포인트 (Custom LangGraph Agent 사용)
# =====================================================
@router.post("/dispatch", response_model=QueryResponse)
async def dispatch(req: QueryRequest, agent: AgentDep):
    """
    [메인 디스패치] 쿼리 라우터 기반 자동 Tool 선택
    
    실행 흐름:
    1. 사용자 쿼리 수신
    2. Session ID 설정 (MCP ToolNode 주입용)
    3. LLM 에이전트 실행 (System Prompt + User Query)
    4. 결과 파싱 및 반환
    """
    print(f"\n[/dispatch] 요청: {req.query}")

    # -----------------------------------------------------
    # 1. Config 구성 (SessionInjectingToolNode 전달용)
    # -----------------------------------------------------
    # RunnableConfig는 딕셔너리 형태로 전달하는 것이 가장 안전합니다.
    # req.session_id가 UUID 타입일 경우 JSON 직렬화를 위해 str() 변환이 필수입니다.
    configurable = {}
    if req.session_id:
        configurable["session_id"] = str(req.session_id)
    
    run_config = {"configurable": configurable}

    # -----------------------------------------------------
    # 2. 메시지 구성 (System Prompt + User Query)
    # -----------------------------------------------------
    # 커스텀 에이전트는 state["messages"]를 입력으로 받습니다.
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=req.query)
    ]
    
    # 향후 대화 기록(History)이 필요하다면 여기서 messages 중간에 삽입합니다.
    if req.session_id:
        history = await get_history(req.session_id)
        if history:
            messages[1:1] = history

    # -----------------------------------------------------
    # 3. 에이전트 실행 (ainvoke)
    # -----------------------------------------------------
    try:
        # agent는 create_custom_agent로 생성된 CompiledGraph입니다.
        result = await agent.ainvoke(
            input={"messages": messages},
            config=run_config
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 실제 운영 환경에서는 에러 메시지를 정제해서 보내는 것이 좋습니다.
        raise HTTPException(status_code=500, detail=f"Agent Execution Failed: {str(e)}")

    # -----------------------------------------------------
    # 4. 결과 처리 및 응답 반환
    # -----------------------------------------------------
    # LangGraph의 결과는 StateDict 형태이며, "messages" 키에 전체 대화 흐름이 담깁니다.
    state_messages: List[BaseMessage] = result.get("messages", [])
    
    # 로그 출력 (디버깅용)
    _log_agent_trace(state_messages)
    
    # 마지막 AI 답변 추출
    ai_text = pick_last_ai_text(state_messages) or ""
    
    # 툴 실행 결과 파싱 (필요 시 클라이언트에 전달)
    tool_response = parse_tool_from_messages(state_messages)

    if req.session_id:
        await persist_history(req.session_id, state_messages)

    return QueryResponse(
        answer=ai_text,
        tool_response=tool_response
    )
