# --- LangChain + Ollama ---
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

# --- FastMCP Client ---
from fastmcp import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport, SSETransport
import asyncio
import json

from core.config import settings


MCP_URL = settings.MCP_SERVER_URL # http://[MCP_SERVER_IP]:[MCP_SERVER_PORT]/mcp
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL_NAME = settings.OLLAMA_MODEL_NAME

# LangChain Tool: 메서드 내부에서 MCP의 tool 호출
class MCPWeatherTool(BaseTool):
    name: str = "get_weather" # MCP 서버의 FastAPI route operation_id와 일치시킴
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
        
        print(f"--- [MCPTool] Connecting to MCP Server at: {MCP_URL}") # URL 확인
        
        # transport = StreamableHttpTransport(url=MCP_URL)
        transport = SSETransport(url=MCP_URL)
        async with MCPClient(transport) as client:
            try:
                # 도구 이름은 FastAPI route의 operation_id에 의해 "get_weather" 로 노출됨
                # (FastMCP 문서: FastAPI operation_id가 MCP 컴포넌트 이름이 됨) :contentReference[oaicite:3]{index=3}
                print(f"--- [MCPTool] Calling tool 'get_weather' with city: {city} ---")
                result = await client.call_tool(self.name, {"city": city})
                # FastMCP 결과 객체 → JSON 직렬화
                print(f"--- [MCPTool] Raw Result Object ---\n{result}") # 디버깅 로그

                # result.content가 리스트이고, 비어있지 않은지 확인
                if result.content and isinstance(result.content, list) and len(result.content) > 0:
                    # content 리스트의 첫 번째 항목(TextContent)에서 .text 속성을 추출
                    data_str = result.content[0].text
                    
                    print(f"--- [MCPTool] Extracted data string from result.content: {data_str}")
                    
                    # AgentRunner가 json.loads() 할 수 있도록 JSON "문자열"을 그대로 반환
                    return data_str
                
                else:
                    # 예상치 못한 응답 (데이터가 없는 경우)
                    print("--- [MCPTool] ERROR: No data found in result.content ---")
                    return "null" # AgentRunner가 'null'을 처리하도록 함
            except Exception as e:
                print(f"--- [MCPTool] ERROR: client.call_tool failed! ---")
                print(e)
                # 에러를 다시 raise해야 상위 BaseTool이 처리합니다.
                raise e

# LLM (LangChain ChatOllama)
llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    request_timeout=300.0
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

def _format_weather_response(city: str, weather_payload: str, base_answer: str = "") -> str:
    """날씨 도구 응답을 사람이 읽기 쉬운 형태로 조합한다."""
    try:
        # 1. weather_payload는 JSON 문자열 (예: '{"city": "Seoul", ...}')입니다.
        data = json.loads(weather_payload)
        
        if isinstance(data, dict):
            pieces = []
            
            resolved_city = data.get("city") or city
            temperature = data.get("temperature_c") 
            conditions = data.get("weather")
            
            if resolved_city and temperature is not None:
                pieces.append(f"{resolved_city}의 현재 기온은 {temperature}°C 입니다.")
            
            if conditions:
                pieces.append(f"날씨 상태: {conditions}")
            
            # 파싱은 성공했으나 유효한 데이터가 없는 경우 원본 JSON 반환
            if not pieces:
                pieces.append(weather_payload)
                
            weather_text = " ".join(pieces)
            
        else:
            # JSON이긴 하지만 dict가 아닌 경우 (예: "null"이 파싱되어 None이 됨)
            weather_text = weather_payload
            
    except json.JSONDecodeError:
        # JSON 파싱 자체에 실패한 경우 (예: "Unknown tool" 같은 에러 메시지)
        weather_text = weather_payload

    if base_answer:
        return f"{base_answer}\n\n{weather_text}"
    return weather_text


class AgentRunner:
    """LangChain 네이티브 툴 호출을 사용하는 새 실행기."""
    def __init__(self, llm_with_tools: ChatOllama, tool: BaseTool):
        self._llm_with_tools = llm_with_tools
        self._tool = tool # MCPWeatherTool 인스턴스

    def run(self, prompt: str) -> str:
        # 시스템 프롬프트를 JSON 강제 대신 일반적인 지시로 변경
        system_message = "You are a helpful assistant. Use the get_weather tool when asked for the weather."
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt),
        ]
        
        print(f"--- [AgentRunner] 1. Calling LLM with tools bound ---")
        # 1. 툴을 사용할지 여부와 함께 LLM 호출
        response = self._llm_with_tools.invoke(messages)

        # 2. 툴 호출 결정 확인
        # (response.tool_calls가 비어있지 않고 내용이 있으면)
        if response.tool_calls:
            print(f"--- [AgentRunner] 2. LLM decided to use tool ---")
            print(response.tool_calls)
            
            tool_call = response.tool_calls[0] # 첫 번째 툴 호출 사용
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")

            if tool_name == self._tool.name and "city" in tool_args:
                city = tool_args["city"]
                print(f"--- [AgentRunner] 3. Calling tool: {tool_name}(city={city}) ---")
                try:
                    # MCPWeatherTool의 _run 메서드 호출 (이 안에서 MCP 서버로 요청)
                    weather_raw = self._tool.run(tool_args["city"])
                    print(f"--- [AgentRunner] 4. Tool Raw Output ---")
                    print(weather_raw)
                    return _format_weather_response(city, weather_raw)
                except Exception as e:
                    print(f"--- [AgentRunner] ERROR: Tool call failed! ---")
                    print(e)
                    # 여기서 에러가 발생한다면, 다음 단계인 URL 문제를 확인해야 합니다.
                    return f"{city} 날씨 조회 중 오류 발생: {e}"
            else:
                return "날씨 도구를 호출하려 했으나, 도시 정보가 누락되었습니다."
        
        # 3. 툴 호출이 없는 경우 (일반 답변)
        print(f"--- [AgentRunner] 2. LLM responded directly ---")
        return str(response.content)

llm_with_tools = llm.bind_tools([weather_tool])
weather_agent = AgentRunner(llm_with_tools=llm_with_tools, tool=weather_tool)
