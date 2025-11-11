from fastapi import FastAPI
from contextlib import asynccontextmanager

from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import settings


async def create_agent_app() -> Runnable:
    """LangChain create_agent+ MCP 도구 구성"""
    
    # MCP 클라이언트 초기화
    try:
        client = MultiServerMCPClient({
            "fisa-mcp": {
                "url": settings.MCP_SERVER_URL,
                "transport": "sse"
            }
        })
        print("✅ MCP 클라이언트가 초기화되었습니다.")
    except Exception as e:
        print(f"⚠️ MCP 클라이언트 초기화 실패: {e}")
        client = None
    
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3,
        request_timeout=300.0
    )

    # MCP 서버 도구를 로드하여 합치기 (이름 중복 제거)
    tools = []
    if client:
        try:
            loaded = await client.get_tools()
            loaded = loaded or []
            existing = {getattr(t, "name", None) for t in tools}
            for t in loaded:
                if getattr(t, "name", None) not in existing:
                    tools.append(t)
            if tools:
                print(f"🔧 사용 도구: {[t.name for t in tools]}")
            else:
                print("⚠️ 사용할 도구가 없습니다.")
        except Exception as e:
            print(f"❌ MCP 서버 도구 로드 실패: {e}")
    else:
        print("⚠️ MCP 클라이언트가 없습니다. retriever_tool만 사용합니다.")

    # 주의: 설치된 langgraph 버전에 따라 state_modifier 인자를 지원하지 않을 수 있음
    # 해당 경우, SYSTEM_PROMPT를 호출부(main.py)에서 SystemMessage로 prepend 하세요.
    agent = create_agent(llm, tools)
    return agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 수명 주기 관리"""
    # 시작 시점: 에이전트 앱 생성
    agent_instance = await create_agent_app()
    app.state.agent = agent_instance
    print("🚀 LLM쿼리 라우팅 서버가 시작되었습니다.")
    yield
    # 종료 시점: 정리 작업 (필요 시 추가)
    print("🛑 LLM쿼리 라우팅 서버가 종료되었습니다.")