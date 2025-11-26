"""간단한 세션별 인메모리 히스토리 저장소."""

from __future__ import annotations

from typing import Dict, List, Sequence

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

_store: Dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """세션별 ChatMessageHistory를 생성/재사용한다."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def get_history(session_id: str) -> BaseChatMessageHistory:
    """RunnableWithMessageHistory에서 직접 사용할 수 있는 헬퍼."""
    return _get_session_history(session_id)


async def get_history(session_id: str) -> List[BaseMessage]:
    """FastAPI 라우터에서 과거 메시지 리스트가 필요할 때 사용."""
    if not session_id:
        return []
    history = _get_session_history(session_id)
    return list(history.messages)


async def persist_history(session_id: str, messages: Sequence[BaseMessage]) -> None:
    """세션별 기록을 현재 LangGraph 결과로 교체한다."""
    if not session_id or not messages:
        return
    _store[session_id] = ChatMessageHistory(messages=list(messages))
