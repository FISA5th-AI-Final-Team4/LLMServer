from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)

from typing import List, Optional, Any


def pick_last_ai_text(messages: List[BaseMessage]) -> Optional[str]:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            return getattr(m, "content", "")
    return None

def _stringify_content(content: Any) -> str:
    """LangChain 메시지 content는 str, dict, list 등 다양할 수 있어 안전하게 문자열화합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                # text 타입을 우선 사용하고, 없으면 dict 전체를 문자열화
                text_value = item.get("text") if "text" in item else None
                if text_value:
                    parts.append(str(text_value))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " | ".join(parts)
    if content is None:
        return ""
    return str(content)

def _log_agent_trace(messages: List[BaseMessage]) -> None:
    """에이전트 사고 흐름과 Tool 사용을 stdout으로 출력합니다."""
    print("[/dispatch] === Agent Trace Start ===")
    if not messages:
        print("[/dispatch] (agent returned no messages)")
        return

    for idx, message in enumerate(messages, start=1):
        role = getattr(message, "type", message.__class__.__name__)
        content = _stringify_content(getattr(message, "content", ""))
        print(f"[step {idx}] role={role}")

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tc_idx, tool_call in enumerate(tool_calls, start=1):
                # tool_call은 dict 혹은 객체일 수 있으므로 안전하게 접근
                name = getattr(tool_call, "name", None)
                if name is None and isinstance(tool_call, dict):
                    name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                args = getattr(tool_call, "args", None)
                if args is None and isinstance(tool_call, dict):
                    args = tool_call.get("args") or tool_call.get("function", {}).get("arguments")
                print(f"        ↪ tool_call[{tc_idx}] {name or 'unknown_tool'} args={args}")

        # ToolMessage에는 tool_call_id / name이 포함되므로 별도 출력
        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = getattr(message, "name", None)
        if tool_call_id or tool_name:
            print(f"        ↩ tool_response from {tool_name or 'unknown_tool'} (call_id={tool_call_id})")

        if content.strip():
            print(f"        content: {content}")

    print("[/dispatch] === Agent Trace End ===\n")