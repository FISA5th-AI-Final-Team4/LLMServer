# --- LangChain + Ollama ---
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

# --- FastMCP Client ---
from fastmcp import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport, SSETransport

import json
import httpx
import asyncio
from typing import List, Dict, Any, AsyncGenerator

from core.config import settings


MCP_URL = settings.MCP_SERVER_URL # http://[MCP_SERVER_IP]:[MCP_SERVER_PORT]/mcp
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL_NAME = settings.OLLAMA_MODEL_NAME

# --- 1. JSON Tool (날씨) ---
class MCPWeatherTool(BaseTool):
    name: str = "get_weather" # MCP 서버의 FastAPI route operation_id와 일치시킴
    description: str = (
        "특정 도시(city)의 현재 날씨나 기온을 조회합니다. "
        "인자: 'city' (string). "
        "예시: 사용자가 '런던의 날씨'라고 말하면, 'city' 인자는 '런던' (조사 '의' 제외)이어야 합니다. "
        "사용자가 '서울 날씨'라고 말하면, 'city' 인자는 '서울'입니다. "
        "반드시 명사 형태의 도시 이름만 추출해야 합니다."
    )

    def _run(self, city: str):
        """LangChain 동기 도구 인터페이스를 위해 비동기 메서드를 실행한다."""
        # 동기 래핑: LangChain Tool은 sync 호출을 기대하므로 내부에서 asyncio 실행
        return asyncio.run(self._arun(city=city))

    async def _arun(self, city: str):
        """FastMCP 서버로 날씨 도구를 호출해 JSON 문자열을 반환한다."""

        print(f"--- [MCPTool-JSON] Connecting to MCP Server at: {MCP_URL}") # URL 확인

        # transport = StreamableHttpTransport(url=MCP_URL)
        transport = SSETransport(url=MCP_URL)
        async with MCPClient(transport) as client:
            try:
                # 도구 이름은 FastAPI route의 operation_id에 의해 "get_weather" 로 노출됨
                # (FastMCP 문서: FastAPI operation_id가 MCP 컴포넌트 이름이 됨) :contentReference[oaicite:3]{index=3}
                print(f"--- [MCPTool-JSON] Calling tool 'get_weather' with city: {city} ---")
                result = await client.call_tool(self.name, {"city": city})
                # FastMCP 결과 객체 → JSON 직렬화
                print(f"--- [MCPTool-JSON] Raw Result Object ---\n{result}")

                # result.content가 리스트이고, 비어있지 않은지 확인
                if result.content and isinstance(result.content, list) and len(result.content) > 0:
                    # content 리스트의 첫 번째 항목(TextContent)에서 .text 속성을 추출
                    data_str = result.content[0].text

                    print(f"--- [MCPTool-JSON] Extracted data string: {data_str}")
                    
                    # AgentRunner가 json.loads() 할 수 있도록 JSON "문자열"을 그대로 반환
                    return data_str
                
                else:
                    # 예상치 못한 응답 (데이터가 없는 경우)
                    print("--- [MCPTool-JSON] ERROR: No data found in result.content ---")
                    return json.dumps({"error": "No data found"})
            except Exception as e:
                print(f"--- [MCPTool-JSON] ERROR: client.call_tool failed! ---")
                print(e)
                error_payload = {"error": f"Tool call failed: {str(e)}"}
                return json.dumps(error_payload)

# --- 2. RAG Streaming Tool ---
class MCPRAGTool(BaseTool):
    name: str = "get_rag_response"  # MCP 서버의 FastAPI route operation_id와 일치시킴
    description: str = (
        "사용자가 '카드 추천', '카드 혜택', '카드 신청' 또는 '금융 용어 설명' 등 "
        "'카드'나 '금융 용어'와 관련된 질문을 할 때만 사용합니다."
        "사용자의 질문을 그대로 인자(query)로 넘긴다."
    )

    def _run(self, query: str):
        """동기 호출은 지원하지 않음."""
        raise NotImplementedError("RAG tool only supports async streaming")

    async def _arun(self, query: str) -> AsyncGenerator[str, None]:
        """FastMCP 서버로 RAG 도구를 호출해 스트리밍 응답을 비동기로 생성한다."""

        rag_endpoint_url = f"{MCP_URL}/tools/rag-generation" 
        
        print(f"--- [MCPTool-RAG-Httpx] Connecting to SSE: {rag_endpoint_url}")

        async with httpx.AsyncClient() as client:
            try:
                print(f"--- [MCPTool-RAG-Httpx] Calling tool 'get_rag_response' with query: {query} ---")
                async with client.stream(
                    "POST", 
                    rag_endpoint_url, 
                    json={"query": query}, 
                    timeout=None
                ) as response:
                    
                    response.raise_for_status() 
                    
                    # SSE 스트림 처리
                    async for line in response.aiter_lines():
                        if not line.strip(): # 빈 줄 무시
                            continue 

                        if line.startswith("data:"): # 데이터 이벤트 처리
                            # "data:" 접두사 제거
                            data_field = line[len("data:"):]

                            if data_field.startswith(' '): # 앞의 공백 제거
                                token = data_field[1:]
                            else: # 공백이 없으면 그대로 사용
                                token = data_field
                            
                            # 스트림 종료 신호 처리
                            if token == "[END_OF_RAG_STREAM]":
                                print("--- [MCPTool-RAG-Httpx] Stream end signal received ---")
                                break
                            
                            # 실제 토큰 생성 부분
                            if token:
                                yield token
                        
                        elif line.startswith("event:"): # 이벤트 타입 처리 (필요시)
                            pass 

            except httpx.HTTPStatusError as e:
                print(f"--- [MCPTool-RAG-Httpx] ERROR: HTTP Status Error {e.response.status_code} ---")
                yield f"[RAG 도구 오류: MCP 서버 응답 실패 ({e.response.status_code})]"
            except httpx.RequestError as e:
                print(f"--- [MCPTool-RAG-Httpx] ERROR: Request Error {e} ---")
                yield f"[RAG 도구 오류: MCP 서버 연결 실패 ({e})]"
            except Exception as e:
                print(f"--- [MCPTool-RAG-Httpx] ERROR: Unknown error {e} ---")
                yield f"[RAG 도구 오류: {e}]"


# --- 3. LLM 및 Tool 인스턴스 생성 ---
llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    request_timeout=300.0
)

# Tool 인스턴스
weather_tool = MCPWeatherTool()
rag_tool = MCPRAGTool()

# --- 4. 응답 포매터 (기존 코드) ---
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

# --- 5. AgentRunner ---
# (시스템 프롬프트 수정)
class AgentRunner:
    def __init__(self, llm: ChatOllama, tools: List[BaseTool]):
        self._llm = llm # 기본 LLM (직접 답변 스트리밍용)
        self._tools = {tool.name: tool for tool in tools} 
        self._llm_with_tools = llm.bind_tools(tools) # Tool 호출 결정용 (라우터 LLM)

    def run(self, prompt: str) -> str:
        """
        [동기식 실행]
        /dispatch-demo/weather 같은 비-스트리밍 엔드포인트에서 호출됩니다.
        LLM의 전체 응답이 완료될 때까지 기다렸다가(blocking), 단일 문자열을 반환합니다.
        """

        # 1. LLM에게 전달할 시스템 프롬프트와 사용자 메시지를 준비합니다.
        #    이 프롬프트는 'get_weather' 도구만 사용하도록 간단하게 지시합니다.
        system_message = "You are a helpful assistant. Use the get_weather tool when asked for the weather."
        messages = [ SystemMessage(content=system_message), HumanMessage(content=prompt) ]

        print(f"--- [AgentRunner.run] 1. Calling LLM (sync) ---")
        
        # 2. 'invoke'를 사용하여 LLM을 동기식으로 호출합니다.
        #    이 호출은 LLM이 Tool 사용을 결정하거나, 직접 답변을 생성할 때까지 완료되지 않습니다.
        response = self._llm_with_tools.invoke(messages)

        # 3. LLM의 응답에 'tool_calls'가 있는지 확인합니다. (Tool 사용 결정 여부)
        if response.tool_calls:
            # 3a. LLM이 Tool을 사용하기로 결정한 경우
            print(f"--- [AgentRunner.run] 2. LLM decided to use tool ---")

            # LLM이 반환한 Tool 호출 정보(이름, 인자)를 추출합니다.
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")

            # 4. Tool 이름과 인자가 유효한지 검사합니다.
            if tool_name == "get_weather" and "city" in tool_args:
                city = tool_args["city"]
                print(f"--- [AgentRunner.run] 3. Calling tool: {tool_name}(city={city}) ---")

                try:
                    # 5. 실제 Tool(.run() 메서드)을 동기적으로 실행합니다.
                    #    (내부적으로 MCPWeatherTool._run -> _arun -> MCP 서버와 통신)
                    weather_raw = self._tools["get_weather"].run(tool_args["city"])

                    print(f"--- [AgentRunner.run] 4. Tool Raw Output ---")
                    print(weather_raw)
                    
                    # 6. Tool이 반환한 JSON 문자열을 사용자 친화적인 텍스트로 포맷팅하여 반환합니다.
                    return _format_weather_response(city, weather_raw)
                
                except Exception as e:
                    # 5-ex. Tool 실행 중 오류가 발생한 경우
                    print(f"--- [AgentRunner.run] ERROR: Tool call failed! ---")
                    return f"{city} 날씨 조회 중 오류 발생: {e}"
            else:
                # 4-ex. LLM이 Tool을 호출하려 했으나, 인자가 잘못된 경우
                return "날씨 도구를 호출하려 했으나, 도시 정보가 누락되었습니다."
        
        # 3b. LLM이 Tool을 사용하지 않기로 결정한 경우 (일반 답변)
        print(f"--- [AgentRunner.run] 2. LLM responded directly ---")

        # 'response.content'에 포함된 LLM의 직접 답변(문자열)을 반환합니다.
        return str(response.content)

    async def arun(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        [비동기 스트리밍 실행]
        /stream-dispatch 엔드포인트에서 호출되는 메인 비동기 제너레이터입니다.
        LLM 쿼리 라우터를 사용하여 1.날씨(JSON), 2.RAG(스트림), 3.일반(스트림) 중 하나를 실행합니다.
        """

        # 1. LLM 쿼리 라우터에게 전달할 시스템 프롬프트입니다.
        #    이 프롬프트는 LLM이 'get_weather', 'get_rag_response', '직접 답변' 중
        #    하나를 결정하도록 지시합니다.
        system_message = (
            "당신은 사용자 요청을 3가지 경로로 분류하는 라우팅 어시스턴트입니다. 당신의 유일한 임무는 어떤 경로를 택할지 결정하는 것입니다."
            "1. 'get_weather' 도구: 사용자가 특정 도시의 '날씨'나 '기온'을 물어볼 때만 사용하세요."
            "2. 'get_rag_response' 도구: 사용자가 '카드 추천', '카드 혜택', '카드 신청', '금융 용어 설명' 등 '카드'나 '금융 용어'와 관련된 질문을 할 때만 사용하세요."
            "3. 직접 답변: 위 1, 2번에 해당하지 않는 모든 일반 대화, 인사, 잡담의 경우, **어떤 도구도 사용하지 말고** 직접 답변하세요."
        )
        
        # 2. LLM에 전달할 전체 메시지 리스트를 구성합니다.
        messages = [ SystemMessage(content=system_message), HumanMessage(content=prompt) ]

        def _yield_json_line(data: Dict[str, Any]) -> str:
            """
            프론트엔드로 보낼 프로토콜({type: ..., payload: ...})을
            JSON Line (JSON 문자열 + \n) 형태로 만듭니다.
            """
            return json.dumps(data, ensure_ascii=False) + "\n"

        try:
            # --- [병목 지점] ---
            # 3. 'ainvoke' (비-스트리밍 호출)를 사용해 라우터 LLM의 *결정*을 받습니다.
            #    이 호출이 완료될 때까지(즉, qwen이 Tool 사용 여부를 결정할 때까지)
            #    첫 토큰 응답(TTFT)이 지연됩니다.
            print(f"--- [AgentRunner.arun] 1. Calling LLM (async) ---")
            response = await self._llm_with_tools.ainvoke(messages)

            # 4. LLM의 결정(response)에 'tool_calls'가 있는지 확인합니다.
            if response.tool_calls:
                # --- 경로 1: Tool 사용 (날씨 또는 RAG) ---
                print(f"--- [AgentRunner.arun] 2. LLM decided to use tool ---")

                # LLM이 결정한 Tool 정보 추출
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")

                if tool_name == "get_weather":
                    # --- 경로 1a: 날씨 Tool (JSON) ---
                    print(f"--- [AgentRunner.arun] 3a. Calling JSON tool: {tool_name} ---")
                    weather_tool = self._tools["get_weather"]

                    # 날씨 Tool을 비동기로 실행하고, 완료된 JSON '문자열'을 받음
                    weather_raw_str = await weather_tool._arun(**tool_args)
                    weather_payload = json.loads(weather_raw_str)

                    # 'json_data' 타입으로 프로토콜을 *단 한 번* yield
                    yield _yield_json_line({ "type": "json_data", "payload": weather_payload })

                elif tool_name == "get_rag_response":
                    # --- 경로 1b: RAG Tool (스트리밍) ---
                    print(f"--- [AgentRunner.arun] 3b. Calling Streaming tool: {tool_name} ---")
                    rag_tool = self._tools["get_rag_response"]

                    # RAG Tool(_arun)은 비동기 제너레이터이므로, 'async for'로 순회
                    async for token in rag_tool._arun(**tool_args):
                        yield _yield_json_line({ "type": "text_chunk", "payload": token })
                
                else:
                    # 'get_weather', 'get_rag_response' 외의 Tool을 호출하려 할 때
                    raise ValueError(f"알 수 없는 Tool 호출: {tool_name}")

            # 5. LLM이 Tool을 사용하지 않기로 결정한 경우 (직접 답변)
            else:
                # --- 경로 2: 직접 답변 (스트리밍) ---
                print(f"--- [AgentRunner.arun] 2. LLM responding directly (streaming) ---")
                
                # [중요] Tool 결정에 사용된 'messages'를 그대로 사용하여
                # LLM 서버의 기본 LLM('self._llm')을 *스트리밍* 모드로 호출합니다.
                async for chunk in self._llm.astream(messages):
                    if chunk.content:
                        # LLM이 생성하는 토큰을 'text_chunk' 타입으로 *즉시* yield
                        yield _yield_json_line({ "type": "text_chunk", "payload": chunk.content })
        
        except Exception as e:
            # 3, 4, 5번 과정 중 어디서든 예외가 발생하면
            print(f"--- [AgentRunner.arun] ERROR: {e} ---")
            # 'error' 타입으로 프로토콜을 yield
            yield _yield_json_line({ "type": "error", "payload": f"Agent 실행 중 오류: {str(e)}" })
        
        finally:
            # 6. (중요) try/except가 끝나면, 성공/실패 여부와 관계없이
            #    항상 'stream_end' 신호를 yield합니다.
            #    (프론트엔드는 이 신호를 받아 입력창을 다시 활성화합니다.)
            print("--- [AgentRunner.arun] 5. Sending stream end ---")
            yield _yield_json_line({ "type": "stream_end", "payload": "[END_OF_STREAM]" })


# --- 6. Agent 인스턴스 생성 ---
weather_agent = AgentRunner(llm=llm, tools=[weather_tool, rag_tool])