from typing import Annotated
from fastapi import Depends, Request, HTTPException
from langchain_core.runnables import Runnable


def get_agent(request: Request) -> Runnable:
    """request.app.state에 저장된 agent 인스턴스를 반환하는 의존성 함수"""
    if not hasattr(request.app.state, 'agent') or request.app.state.agent is None:
        # lifespan에서 agent가 제대로 로드되지 않았을 경우 오류 발생
        raise HTTPException(status_code=503, detail="Agent가 초기화되지 않았습니다.")
    
    return request.app.state.agent

# --- (Annotated 타입 힌트 정의) ---
AgentDep = Annotated[Runnable, Depends(get_agent)]