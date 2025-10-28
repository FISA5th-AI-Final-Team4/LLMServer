from typing import Any

from fastapi import APIRouter, Body

from schemas.mcp_router import QueryRequest, QueryResponse

from agent.agent import agent

router = APIRouter(prefix="/mcp-router", tags=["Query Routing"])


@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""

    return payload


# @router.post("/invoke-direct")
# async def invoke_direct(prompt: str = Body(..., embed=True)) -> Any:
#     """Response for the prompt directly without dispatching through MCP tools."""
#     # TODO - LLM 연동

#     return {"response": f"Direct response to prompt: {prompt}"}


@router.post("/invoke", response_model=QueryResponse)
def chat(req: QueryRequest):
    """
    사용자의 질문을 받아, 에이전트가 날씨가 필요하면 MCP Weather Tool(get_weather)을 호출,
    아니면 LLM 자체 답변을 반환.
    """
    # 프롬프트 힌트: 도구 호출 기준을 LLM에 분명히 안내
    system_hint = (
        "당신은 도우미입니다. 사용자가 특정 도시의 현재 날씨/기온을 묻는다면 "
        "'get_weather' 도구를 반드시 호출해 답을 구성하세요. "
        "그 외 일반 질문은 도구 없이 직접 답하세요. "
        "도구 인자는 city (예: 'Seoul', 'New York')."
    )
    # LangChain의 ReAct 프롬프트에 힌트를 주기 위해 입력 앞에 붙입니다.
    user_q = f"{system_hint}\n\n질문: {req.query}"

    result = agent.run(user_q)
    # weather_tool가 반환한 JSON 문자열을 에이전트가 그대로 포함할 수 있으므로,
    # 응답 가공(선택) — 간단히 그대로 반환
    return QueryResponse(answer=result)


@router.post("/dispatch")
async def dispatch():
    """Placeholder dispatch endpoint that simply returns the received payload."""
    # TODO - MCP 도구 연동 및 디스패치 로직 구현

    return {"message": "Dispatch endpoint is under construction."}
