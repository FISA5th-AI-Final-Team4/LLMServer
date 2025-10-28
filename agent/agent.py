# --- LangChain + Ollama ---
from langchain_community.chat_models import ChatOllama
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

# --- FastMCP Client (원격 MCP 접속) ---
from fastmcp import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport
import asyncio
import json

from core.config import settings


# MCP 서버의 MCP 엔드포인트 (호스트 머신의 MCP 서비스 사용)
MCP_URL = settings.MCP_SERVER_URL
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL_NAME = settings.OLLAMA_MODEL_NAME

# LangChain Tool: 메서드 내부에서 MCP의 tool 호출
class MCPWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = (
        "도시명을 받아 현재 기온(섭씨)을 조회한다. "
        "사용자는 한국어로 물어볼 수 있다. 인자: city (string)"
    )

    def _run(self, city: str):
        """LangChain 동기 도구 인터페이스를 위해 비동기 메서드를 실행한다."""
        # 동기 래핑: LangChain Tool은 sync 호출을 기대하므로 내부에서 asyncio 실행
        return asyncio.run(self._arun(city=city))

    async def _arun(self, city: str):
        """FastMCP 서버로 날씨 도구를 호출해 JSON 문자열을 반환한다."""
        transport = StreamableHttpTransport(url=MCP_URL)
        async with MCPClient(transport) as client:
            # 도구 이름은 FastAPI route의 operation_id에 의해 "get_weather" 로 노출됨
            # (FastMCP 문서: FastAPI operation_id가 MCP 컴포넌트 이름이 됨) :contentReference[oaicite:3]{index=3}
            result = await client.call_tool("get_weather", {"city": city})
            # FastMCP 결과 객체 → JSON 직렬화
            return json.dumps(result.data, ensure_ascii=False)

# LLM (LangChain ChatOllama)
llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
)

# Tool 인스턴스
weather_tool = MCPWeatherTool()

DECISION_SYSTEM_PROMPT = """
You are a routing assistant that decides whether a user's request needs real-time weather data.
Respond strictly with a minified JSON object that follows this schema:
{
  "use_tool": <true|false>,
  "city": "<city name or empty string>",
  "answer": "<concise natural language answer if no tool is needed>"
}
- Set "use_tool" to true only when the user asks for weather or temperature for a specific place.
- When "use_tool" is true, fill "city" with the location name (in English if possible) and leave "answer" as an empty string.
- When "use_tool" is false, ignore "city" and provide the direct reply in "answer".
Only output JSON, with no additional text.
""".strip()


def _call_llm(prompt: str) -> str:
    """LLM을 호출해 텍스트 응답을 문자열로 추출한다."""
    messages = [
        SystemMessage(content=DECISION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "".join(chunks)
    return str(content)


def _parse_decision(raw_text: str) -> dict[str, str | bool]:
    """LLM이 반환한 JSON 결정을 파싱하고 실패 시 안전한 기본값을 제공한다."""
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Decision payload is not a JSON object")
        return data
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat as direct answer
        return {"use_tool": False, "city": "", "answer": raw_text}


def _format_weather_response(city: str, weather_payload: str, base_answer: str = "") -> str:
    """날씨 도구 응답을 사람이 읽기 쉬운 형태로 조합한다."""
    try:
        data = json.loads(weather_payload)
        if isinstance(data, dict):
            pieces = []
            resolved_city = data.get("city") or city
            temperature = data.get("temperature")
            conditions = data.get("conditions") or data.get("description")
            if resolved_city and temperature is not None:
                pieces.append(f"{resolved_city}의 현재 기온은 {temperature}°C 입니다.")
            if conditions:
                pieces.append(f"날씨 상태: {conditions}")
            if not pieces:
                pieces.append(weather_payload)
            weather_text = " ".join(pieces)
        else:
            weather_text = weather_payload
    except json.JSONDecodeError:
        weather_text = weather_payload

    if base_answer:
        return f"{base_answer}\n\n{weather_text}"
    return weather_text


class AgentRunner:
    """기존 `.run()` 메서드 형태를 유지하는 호환 레이어."""

    def __init__(self, tool: BaseTool):
        """사용할 MCP 도구 인스턴스를 보관한다."""
        self._tool = tool

    def run(self, prompt: str) -> str:
        """프롬프트를 받아 도구 사용 여부를 판단하고 최종 답변을 생성한다."""
        decision_raw = _call_llm(prompt)
        decision = _parse_decision(decision_raw)

        if decision.get("use_tool") and decision.get("city"):
            city = str(decision["city"])
            weather_raw = self._tool.run(city)  # synchronous wrapper
            base_answer = str(decision.get("answer") or "").strip()
            return _format_weather_response(city, weather_raw, base_answer)

        answer = str(decision.get("answer") or "").strip()
        return answer or decision_raw


# Backwards-compatible export
agent = AgentRunner(weather_tool)
