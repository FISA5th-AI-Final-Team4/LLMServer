import operator
from typing import Annotated, List, TypedDict, Union, Any, Set

from fastapi import FastAPI, Depends, Request, HTTPException
from contextlib import asynccontextmanager

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from core.config import settings

# =========================================================
# 0. State & Custom Node 정의
# =========================================================

class AgentState(TypedDict):
    """에이전트 상태 정의 (메시지 누적)"""
    messages: Annotated[List[BaseMessage], add_messages]

# =========================================================
# 1. Fallback Node 정의 (기본 응답 생성)
# =========================================================
async def fallback_node(state: AgentState, config: RunnableConfig):
    """
    툴이 선택되지 않았을 때 실행되는 노드.
    기존 LLM의 응답(잡담 등)을 무시하고, 정해진 Default String을 반환합니다.
    """
    default_msg = "죄송합니다! 그 부분은 더 공부해올게요!\n😊 우리카드 관련 질문이 있으시면 언제든 물어보세요."
    
    # 마지막 메시지(LLM의 잡담)를 덮어쓰거나, 새로운 메시지로 추가합니다.
    # 여기서는 명확한 처리를 위해 AIMessage로 반환합니다.
    return {"messages": [AIMessage(content=default_msg)]}

# =========================================================
# 2. 커스텀 라우팅 로직 (Tool 선택 여부 판단)
# =========================================================
def route_decision(state: AgentState):
    """
    마지막 메시지를 확인하여 다음 단계 결정
    - tool_calls가 있으면 -> 'tools' 노드로
    - 없으면 -> 'fallback' 노드로
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return "fallback"

# =========================================================
# 3. 커스텀 ToolNode 정의 (Session ID 주입)
# =========================================================
class SessionInjectingToolNode(ToolNode):
    def __init__(self, tools: List[BaseTool], **kwargs: Any):
        super().__init__(tools, **kwargs)

        # 1) args_schema에 session_id가 있는 툴만 따로 기록
        self.tools_requiring_session_id: Set[str] = set()
        for t in tools:
            name = getattr(t, "name", None)
            schema = getattr(t, "args_schema", None)
            
            if not name or not schema:
                continue

            has_session_id = False

            # [Case A] Pydantic V2 (model_fields 사용)
            if hasattr(schema, "model_fields"):
                if "session_id" in schema.model_fields:
                    has_session_id = True
            
            # [Case B] Pydantic V1 (__fields__ 사용)
            elif hasattr(schema, "__fields__"):
                if "session_id" in schema.__fields__:
                    has_session_id = True
            
            # [Case C] 딕셔너리 형태인 경우 (일부 커스텀 툴)
            elif isinstance(schema, dict):
                props = schema.get("properties", {})
                if "session_id" in props:
                    has_session_id = True

            if has_session_id:
                self.tools_requiring_session_id.add(name)

        print(
            "[SessionInjectingToolNode] session_id 필드가 있는 MCP 툴 목록:",
            self.tools_requiring_session_id,
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        # ---- 디버깅용 로그 ----
        print("\n[SessionInjectingToolNode] ainvoke 호출됨")
        print(f"[SessionInjectingToolNode] input 타입: {type(input)}")
        # print(f"[SessionInjectingToolNode] config: {config}")

        # 2) config에서 session_id 추출
        session_id = None
        if isinstance(config, dict):
            session_id = config.get("configurable", {}).get("session_id")

        if not session_id:
            print(
                "[SessionInjectingToolNode] config.configurable.session_id 없음 → 인젝션 스킵"
            )
            return await super().ainvoke(input, config=config, **kwargs)

        # 3) UUID 형식 검증 (보장용)
        import uuid

        try:
            session_id = str(uuid.UUID(str(session_id)))
        except Exception:
            print(
                f"[SessionInjectingToolNode] 잘못된 session_id 형식: {session_id} → 인젝션 스킵"
            )
            return await super().ainvoke(input, config=config, **kwargs)

        print(f"[SessionInjectingToolNode] 사용할 session_id: {session_id}")

        # 4) 그래프 state(딕셔너리) 형태만 처리: {"messages": [...]}
        if isinstance(input, dict) and "messages" in input:
            messages = list(input["messages"])
            if not messages:
                print("[SessionInjectingToolNode] messages 비어 있음 → 인젝션 스킵")
                return await super().ainvoke(input, config=config, **kwargs)

            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", None)

            if not tool_calls:
                print("[SessionInjectingToolNode] 마지막 메시지에 tool_calls 없음 → 인젝션 스킵")
                return await super().ainvoke(input, config=config, **kwargs)

            print(
                f"[SessionInjectingToolNode] 마지막 AIMessage에 tool_calls {len(tool_calls)}개 발견"
            )

            # 5) 각 tool_call에 대해, session_id 필드가 필요한 MCP 툴에만 인젝션
            for call in tool_calls:
                # call은 dict 또는 dataclass일 수 있음
                tool_name = getattr(call, "name", None)
                if not tool_name and isinstance(call, dict):
                    tool_name = call.get("name")

                print(f"[SessionInjectingToolNode] 처리 중인 tool_call name={tool_name}")

                if tool_name not in self.tools_requiring_session_id:
                    # session_id가 필요 없는 툴은 건드리지 않음
                    continue

                # 기존 session_id를 덮어쓰는 로직
                if isinstance(call, dict):
                    args = dict(call.get("args") or {})
                    args["session_id"] = session_id  # 기존 값 덮어쓰기
                    call["args"] = args
                elif hasattr(call, "args"):
                    args = dict(getattr(call, "args") or {})
                    args["session_id"] = session_id  # 기존 값 덮어쓰기
                    setattr(call, "args", args)
                else:
                    print(
                        f"[SessionInjectingToolNode] 예상치 못한 tool_call 구조: {type(call)} → 인젝션 실패"
                    )
                    continue

                print(
                    f"[SessionInjectingToolNode] '{tool_name}' 툴에 session_id 인젝션 완료: {session_id}"
                )

            # 수정된 messages를 다시 state에 넣어서 super로 넘김
            input = {**input, "messages": messages}
        else:
            print(
                "[SessionInjectingToolNode] input이 dict(messages 포함)이 아님 → 인젝션 스킵"
            )

        # 6) 원래 ToolNode 로직 수행
        return await super().ainvoke(input, config=config, **kwargs)


# =========================================================
# 4. Agent 생성 로직 (Workflow 조립)
# =========================================================

def create_custom_agent(llm, tools: List[BaseTool]) -> Runnable:
    """
    LLM과 Tools를 받아 SessionInjectingToolNode가 적용된 Compiled Graph를 반환
    ※ 사용자 메모리(Checkpointer)는 포함하지 않음 (Stateless)
    """
    
    # 1. LLM에 툴 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 2. 에이전트 노드 (생각하는 단계) 정의
    async def agent_node(state: AgentState, config: RunnableConfig):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"], config)]}

    # 3. Workflow 초기화
    workflow = StateGraph(AgentState)

    # 4. 노드 추가
    workflow.add_node("agent", agent_node)
    # ★ 핵심: 커스텀 ToolNode 사용
    workflow.add_node("tools", SessionInjectingToolNode(tools))
    # ★ 기본응답 출력 노드
    workflow.add_node("fallback", fallback_node)

    # 5. 엣지 연결
    workflow.add_edge(START, "agent")
    # workflow.add_conditional_edges("agent", tools_condition) 
    workflow.add_conditional_edges(
        "agent",
        route_decision,
        {
            "tools": "tools",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("tools", END)
    workflow.add_edge("fallback", END)

    # 6. 컴파일 (메모리 없이 컴파일)
    return workflow.compile()


def get_agent(request: Request) -> Runnable:
    """request.app.state에 저장된 agent 인스턴스를 반환하는 의존성 함수"""
    if not hasattr(request.app.state, 'agent') or request.app.state.agent is None:
        # lifespan에서 agent가 제대로 로드되지 않았을 경우 오류 발생
        raise HTTPException(status_code=503, detail="Agent가 초기화되지 않았습니다.")
    
    return request.app.state.agent

# --- (Annotated 타입 힌트 정의) ---
AgentDep = Annotated[Runnable, Depends(get_agent)]