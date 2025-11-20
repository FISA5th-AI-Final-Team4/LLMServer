import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, ToolMessage

JsonValue = Union[Dict[str, Any], List[Any]]

def parse_tool_from_messages(messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
    """
    LangChain agent 호출 결과에서 Tool 응답(JSON)을 추출합니다.

    Returns:
        tool_name, tool_args, tool_response_content을 포함한 dict (없으면 None)
    """
    # 파싱할 메세지가 없으면 조기 반환
    if not messages:
        return None

    tool_metadata: Dict[str, Any] = {} # 반환값 초기화
    last_tool_call_id: Optional[str] = None # 호출 Tool과 응답 페이로드 매칭을 위한 ID

    for message in reversed(messages):
        # 메세지 타입 판별
        role = (getattr(message, "type", None) or message.__class__.__name__ or "").lower()

        # 툴 호출 결과인 경우
        if role == "tool" and "tool_response_content" not in tool_metadata:
            # tool_call_id를 저장하여 이후 AI 메시지에서 호출 정보와 매칭
            last_tool_call_id = getattr(message, "tool_call_id", None)
            # 문자열 형태의 tool 호출 결과를 dict/list로 역직렬화 시도
            parsed_content = _deserialize_tool_content(getattr(message, "content", None))
            if parsed_content is not None:
                tool_metadata["tool_response_content"] = parsed_content
            else:
                tool_metadata["tool_response_content"] = getattr(message, "content", None)
            # operation_id 할당
            tool_metadata["tool_name"] = getattr(message, "name", tool_metadata.get("tool_name"))
            continue

        # ai 메세지 이고 tool_calls가 있는 경우 파싱
        if role == "ai":
            # 하나의 메세지에서 호출한 tool 목록
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in reversed(tool_calls):
                # 호출 id 파싱 (aff3ff30-e108-411d-a361-a219dad2cbd6 형태)
                call_id = _get_tool_call_field(tool_call, "id")
                # 마지막 tool_call_id와 매칭되는 호출만 처리
                if last_tool_call_id and call_id and call_id != last_tool_call_id:
                    continue
                # tool 이름/인자 파싱
                name = _get_tool_call_field(tool_call, "name")
                args = _get_tool_call_args(tool_call)
                if name and "tool_name" not in tool_metadata:
                    tool_metadata["tool_name"] = name
                if args is not None:
                    tool_metadata["tool_args"] = args
                if call_id and "tool_call_id" not in tool_metadata:
                    tool_metadata["tool_call_id"] = call_id
                break

        # human 메세지에 도달하면 종료
        if role == "human" and tool_metadata:
            break

    return tool_metadata or None

def _deserialize_tool_content(content: Any) -> Optional[JsonValue]:
    """
    ToolMessage.content를 dict/list로 역직렬화합니다.

    LangChain ToolMessage는 주로 str JSON을 반환하지만, dict/list가 오는 경우도
    있으므로 최대한 안전하게 처리합니다.
    """
    if content is None:
        return None

    if isinstance(content, dict):
        if _looks_like_text_chunk(content):
            return _loads_json_string(str(content.get("text", "")))
        return content  # Tool이 dict를 바로 반환한 경우

    if isinstance(content, list):
        if _looks_like_text_chunks(content):
            text_payload = "".join(str(part.get("text", "")) for part in content)
            return _loads_json_string(text_payload)
        if len(content) == 1:
            return _deserialize_tool_content(content[0])
        return content  # JSON 배열 그대로 반환

    if isinstance(content, str):
        return _loads_json_string(content)

    # 리스트 안에 조각난 text dict가 올 수 있으므로 문자열화 후 파싱 시도
    if isinstance(content, (tuple, set)):
        joined = "".join(str(part) for part in content)
        return _loads_json_string(joined)

    return _loads_json_string(str(content))


def _loads_json_string(raw: str) -> Optional[JsonValue]:
    """문자열에서 JSON 부분을 찾아 파싱합니다."""
    if not raw:
        return None

    cleaned = _strip_code_fences(raw.strip())
    for candidate in _candidate_json_strings(cleaned):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _strip_code_fences(text: str) -> str:
    """
    ```json ... ``` 형태의 코드 블록이 포함된 경우 내용을 추출합니다.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _candidate_json_strings(text: str) -> List[str]:
    """
    완전한 JSON을 찾기 위해 원본문, {...}, [...] 조각을 순차적으로 반환합니다.
    """
    candidates = [text]
    object_span = _extract_span(text, "{", "}")
    if object_span:
        candidates.append(object_span)
    array_span = _extract_span(text, "[", "]")
    if array_span:
        candidates.append(array_span)
    return candidates


def _extract_span(text: str, start_char: str, end_char: str) -> Optional[str]:
    """주어진 시작/끝 문자 사이의 가장 바깥 JSON 구간을 추출합니다."""
    start = text.find(start_char)
    end = text.rfind(end_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _get_tool_call_field(tool_call: Any, field: str) -> Any:
    """tool_call 객체/딕셔너리에서 원하는 필드를 추출합니다."""
    if isinstance(tool_call, dict):
        if field in tool_call:
            return tool_call[field]
        function = tool_call.get("function")
        if isinstance(function, dict):
            if field == "name":
                return function.get("name")
            if field in {"args", "arguments"}:
                return function.get("arguments")
    value = getattr(tool_call, field, None)
    if value is not None:
        return value
    function = getattr(tool_call, "function", None)
    if isinstance(function, dict):
        if field == "name":
            return function.get("name")
        if field in {"args", "arguments"}:
            return function.get("arguments")
    return None


def _get_tool_call_args(tool_call: Any) -> Any:
    """tool_call args를 dict로 역직렬화합니다."""
    args = _get_tool_call_field(tool_call, "args")
    if args is None:
        args = _get_tool_call_field(tool_call, "arguments")
    if isinstance(args, str):
        parsed = _loads_json_string(args)
        if parsed is not None:
            return parsed
    return args


def _looks_like_text_chunk(value: Any) -> bool:
    """LangChain 메시지 content 조각(dict)에 해당하는지 판별합니다."""
    return (
        isinstance(value, dict)
        and "type" in value
        and "text" in value
        and isinstance(value.get("text"), str)
    )


def _looks_like_text_chunks(value: List[Any]) -> bool:
    return bool(value) and all(_looks_like_text_chunk(item) for item in value)
