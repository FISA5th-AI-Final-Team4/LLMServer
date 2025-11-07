from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from schemas.mcp_router import QueryRequest, QueryResponse
from agent.weather_agent import weather_agent

router = APIRouter(tags=["MCP Client Dispatch"], prefix="/mcp-router")


@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""

    return payload


@router.post("/dispatch-demo/weather", response_model=QueryResponse)
def chat(req: QueryRequest):
    """
    사용자의 질문을 받아, 에이전트가 날씨가 필요하면 MCP Weather Tool(get_weather)을 호출,
    아니면 LLM 자체 답변을 반환. (스트리밍 X)
    """
    # 프롬프트 힌트: 도구 호출 기준을 LLM에 분명히 안내
    system_hint = (
        "당신은 도우미입니다. 사용자가 특정 도시의 현재 날씨/기온을 묻는다면 "
        "'get_weather' 도구를 반드시 호출해 답을 구성하세요. "
        "그 외 일반 질문은 도구 없이 직접 답하세요. "
        "도구 인자는 city (예: 'Seoul', 'New York')."
    )
    user_q = f"{system_hint}\n\n질문: {req.query}"

    # AgentRunner.run() (동기 메서드) 호출
    result = weather_agent.run(user_q)
    
    return QueryResponse(answer=result)


@router.post("/stream-dispatch")
async def stream_dispatch(req: QueryRequest):
    """
    [비동기 스트리밍]
    백엔드 서버로부터 채팅 요청을 받아, AgentRunner.arun() 제너레이터를 호출하고,
    그 결과를 StreamingResponse로 중계(릴레이)합니다.
    
    이 스트림은 {type: ..., payload: ...} 형태의 JSON Line입니다.
    """
    print(f"LLM (Stream): 프롬프트 수신 -> '{req.query}'")
    
    # AgentRunner.arun()은 비동기 제너레이터입니다.
    # StreamingResponse가 이 제너레이터를 순회하며
    # yield된 각 항목(JSON Line)을 백엔드 서버로 전송합니다.
    return StreamingResponse(
        weather_agent.arun(req.query),
        media_type="application/x-json-stream" # JSON Line 스트림
    )


@router.post("/dispatch")
async def dispatch():
    """Placeholder dispatch endpoint that simply returns the received payload."""
    # TODO - MCP 도구 연동 및 디스패치 로직 구현

    return {"message": "Dispatch endpoint is under construction."}