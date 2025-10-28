from typing import Any

from fastapi import APIRouter, Body


router = APIRouter(prefix="/mcp-router", tags=["Query Routing"])


@router.post("/echo")
async def echo(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Echo the incoming payload for quick connectivity checks."""

    return payload


@router.post("/invoke")
async def invoke(prompt: str = Body(..., embed=True)) -> Any:
    """Response for the prompt directly without dispatching through MCP tools."""
    # TODO - LLM 연동

    return {"response": f"Direct response to prompt: {prompt}"}


@router.post("/dispatch")
async def dispatch():
    """Placeholder dispatch endpoint that simply returns the received payload."""
    # TODO - MCP 도구 연동 및 디스패치 로직 구현

    return {"message": "Dispatch endpoint is under construction."}
