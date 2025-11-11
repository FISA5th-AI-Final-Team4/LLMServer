"""
쿼리 라우터 에이전트 (LangChain Tool 기반)

실제 실행 순서:
- 1단계: LangChain Tool 자동 선택 (LLM이 사용자 쿼리 분석)
- 2단계: 선택된 Tool 내부에서 쿼리 전처리 (Query Preprocessing)
- 3단계: 전처리된 쿼리로 MCP 서버 호출 (operation_id 매칭)
"""

from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

import json
import httpx
import asyncio
from typing import Dict, Any, List

from core.config import settings
from schemas.mcp_router import PreprocessedQuery


MCP_URL = settings.MCP_SERVER_URL  # http://[MCP_SERVER_IP]:[MCP_SERVER_PORT]/mcp

# --- LLM 인스턴스 생성 ---
# 전처리용 LLM (정확성 우선)
preprocessing_llm = ChatOllama(
    model=settings.PREPROCESSING_MODEL_NAME or settings.OLLAMA_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=settings.PREPROCESSING_TEMPERATURE,
    request_timeout=60.0
)

# Tool 선택용 LLM (일관성 우선)
tool_selector_llm = ChatOllama(
    model=settings.ROUTING_MODEL_NAME or settings.OLLAMA_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=settings.ROUTING_TEMPERATURE,
    request_timeout=60.0
)


# =====================================================
# 쿼리 전처리 함수 (Tool 내부에서 호출)
# =====================================================

def _preprocess_query_internal(query: str) -> PreprocessedQuery:
    """
    사용자의 원본 쿼리를 벡터 검색에 최적화된 형태로 전처리합니다.
    
    수행 작업:
    - 구어체 → 검색용 표준 표현으로 변환
    - 핵심 키워드 추출
    - 오타 수정 및 약어 해소
    - 불필요한 조사/어미 제거
    
    Args:
        query: 사용자의 원본 쿼리
        
    Returns:
        PreprocessedQuery: 전처리 결과 (정제된 쿼리, 키워드, 신뢰도)
    """
    
    print(f"\n{'='*60}")
    print(f"[1단계: 쿼리 전처리 시작]")
    print(f"원본 쿼리: {query}")
    print(f"{'='*60}")
    
    # 시스템 프롬프트: 전처리 지침
    system_prompt = """당신은 사용자 쿼리를 벡터 검색에 최적화된 형태로 전처리하는 전문가입니다.

**당신의 임무**:
1. 사용자의 구어체 쿼리를 명확하고 검색 가능한 표준 표현으로 변환
2. 핵심 키워드를 3~5개 추출 (카드 혜택, 소비 패턴 관련)
3. 오타 수정, 약어 해소, 불필요한 조사 제거
4. 전처리 결과의 품질을 0.0~1.0으로 평가

**예시**:
입력: "20대 여잔데 쇼핑 자주함"
출력: 
{
  "normalized_query": "20대 여성 쇼핑 할인 카드 추천",
  "key_keywords": ["20대", "여성", "쇼핑", "할인", "카드"],
  "confidence": 0.85
}

입력: "편의점 많이씀 할인되는거"
출력:
{
  "normalized_query": "편의점 할인 카드 추천",
  "key_keywords": ["편의점", "할인", "카드", "추천"],
  "confidence": 0.9
}

**중요 규칙**:
- normalized_query는 반드시 한국어로 작성
- key_keywords는 정확히 3~5개
- confidence는 전처리 품질에 대한 자신감 (높을수록 좋음)
- 응답은 반드시 JSON 형식만 출력 (다른 텍스트 금지)"""

    user_prompt = f"다음 쿼리를 전처리하세요:\n\n{query}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        # LLM 호출
        print(f"[전처리 LLM 호출 중...]")
        response = preprocessing_llm.invoke(messages)
        raw_output = response.content
        
        print(f"[LLM 원본 응답]\n{raw_output}\n")
        
        # JSON 파싱
        # LLM이 ```json ... ``` 형태로 응답할 수 있으므로 처리
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            json_str = raw_output.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_output.strip()
        
        parsed_data = json.loads(json_str)
        
        # PreprocessedQuery 객체 생성
        result = PreprocessedQuery(
            original_query=query,
            normalized_query=parsed_data.get("normalized_query", query),
            key_keywords=parsed_data.get("key_keywords", []),
            confidence=float(parsed_data.get("confidence", 0.5))
        )
        
        print(f"[전처리 완료]")
        print(f"✓ 정제된 쿼리: {result.normalized_query}")
        print(f"✓ 핵심 키워드: {', '.join(result.key_keywords)}")
        print(f"✓ 신뢰도: {result.confidence:.2f}")
        print(f"{'='*60}\n")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 실패: {e}")
        print(f"[폴백] 원본 쿼리 사용")
        
        # 파싱 실패 시 원본 쿼리를 그대로 사용 (신뢰도 낮게)
        return PreprocessedQuery(
            original_query=query,
            normalized_query=query,
            key_keywords=[],
            confidence=0.3
        )
        
    except Exception as e:
        print(f"[오류] 전처리 실패: {e}")
        print(f"[폴백] 원본 쿼리 사용")
        
        return PreprocessedQuery(
            original_query=query,
            normalized_query=query,
            key_keywords=[],
            confidence=0.0
        )


# =====================================================
# MCP Tools 정의 (LangChain Tool 기반)
# - Tool 선택 후, 각 Tool 내부에서 전처리 수행
# - operation_id 자동 매칭으로 MCP 서버 호출
# =====================================================

# --- Tool 1: 카드 추천 RAG Tool ---
class MCPCardRecommendationTool(BaseTool):
    name: str = "get_card_recommendation"  # MCP operation_id와 일치
    description: str = (
        "사용자가 신용카드/체크카드 추천을 요청할 때 사용합니다. "
        "'카드 추천', '할인 카드', '쇼핑 카드', '편의점 카드', '주유 카드' 등의 키워드 포함 시 사용. "
        "사용자의 소비 패턴이나 필요한 혜택을 query로 넘깁니다."
    )
    
    def _run(self, query: str):
        """동기 호출용 (비동기 래핑)"""
        return asyncio.run(self._arun(query=query))
    
    async def _arun(self, query: str) -> str:
        """MCP 서버의 카드 추천 RAG 파이프라인을 호출합니다."""
        
        # Tool 내부 Step 1: 쿼리 전처리 (검색 최적화)
        print(f"--- [CardTool] 1. 쿼리 전처리 시작 ---")
        preprocessed = _preprocess_query_internal(query)
        
        # Tool 내부 Step 2: MCP 서버 호출
        base_url = MCP_URL.replace('/mcp', '')
        endpoint_url = f"{base_url}/tools/card-recommendation"
        
        print(f"--- [CardTool] 2. MCP 서버 호출: {endpoint_url} ---")
        print(f"--- [CardTool] 정제된 쿼리: {preprocessed.normalized_query} ---")
        
        async with httpx.AsyncClient() as client:
            try:
                print(f"--- [CardTool] 요청 데이터: query={preprocessed.normalized_query}, retrieve_k=5, final_k=3 ---")
                response = await client.post(
                    endpoint_url,
                    json={
                        "query": preprocessed.normalized_query,
                        "retrieve_k": 5,  # MCP 서버 최소값
                        "final_k": 3      # 최종 결과
                    },
                    timeout=180.0  # 타임아웃을 180초로 증가
                )
                
                print(f"--- [CardTool] 응답 상태 코드: {response.status_code} ---")
                response.raise_for_status()
                result_data = response.json()
                
                answer = result_data.get("answer", "")
                context_docs = result_data.get("context_docs", [])
                
                # 답변 포맷팅
                formatted_response = answer
                if context_docs:
                    formatted_response += "\n\n📋 참고 카드:"
                    for i, doc in enumerate(context_docs, 1):
                        card_name = doc.get("metadata", {}).get("card_name", "알 수 없음")
                        formatted_response += f"\n{i}. {card_name}"
                
                print(f"--- [CardTool] 3. 응답 완료 (길이: {len(formatted_response)}자) ---")
                return formatted_response
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"--- [CardTool] ERROR: {e} ---")
                print(f"--- [CardTool] ERROR 상세:\n{error_detail} ---")
                return f"[카드 추천 오류: {e}]"


# --- Tool 2: ML 기반 소비 패턴 분석 Tool ---
class MCPMyDataAnalysisTool(BaseTool):
    name: str = "analyze_consumption_pattern"  # MCP operation_id와 일치 (예정)
    description: str = (
        "사용자의 소비 패턴을 분석하거나 통계를 제공할 때 사용합니다. "
        "'소비 패턴', '어디에 돈 쓰나', '통계', '분석해줘' 등의 키워드 포함 시 사용. "
        "MyData 기반 개인화된 분석을 제공합니다."
    )
    
    def _run(self, query: str):
        """동기 호출용"""
        return asyncio.run(self._arun(query=query))
    
    async def _arun(self, query: str) -> str:
        """ML 기반 소비 패턴 분석 (미구현)"""
        print(f"--- [MyDataTool] 호출: {query} ---")
        return "[소비 패턴 분석 기능은 준비 중입니다. 곧 제공될 예정입니다.]"


# --- Tool 3: QnA DB 조회 Tool ---
class MCPQnADatabaseTool(BaseTool):
    name: str = "query_faq_database"  # MCP operation_id와 일치 (예정)
    description: str = (
        "간단한 정보 조회나 FAQ 질문에 답변할 때 사용합니다. "
        "'이벤트 언제까지', '프로모션', '신청 방법', '고객센터', '영업시간' 등의 키워드 포함 시 사용. "
        "데이터베이스에서 빠르게 정보를 검색합니다."
    )
    
    def _run(self, query: str):
        """동기 호출용"""
        return asyncio.run(self._arun(query=query))
    
    async def _arun(self, query: str) -> str:
        """QnA DB 조회 (미구현)"""
        print(f"--- [QnATool] 호출: {query} ---")
        return "[FAQ 데이터베이스 기능은 준비 중입니다.]"


# 통합 에이전트 (weather_agent.AgentRunner 스타일)
# =====================================================

class QueryRouterAgent:
    """
    LangChain Tool 기반 쿼리 라우터 에이전트
    
    실제 실행 흐름:
    1. 사용자 쿼리 입력 (원본 구어체 그대로)
    2. LLM이 적절한 Tool 자동 선택 (get_card_recommendation, analyze_consumption_pattern 등)
       → Tool 선택은 구어체로 판단 (의도 파악 용이)
    3. 선택된 Tool 내부에서 쿼리 전처리 수행
       → 검색 최적화된 형태로 변환
    4. 전처리된 쿼리로 MCP 서버 호출 (operation_id 자동 매칭)
    5. 결과 반환
    
    사용 예시:
        agent = QueryRouterAgent()
        result = agent.run("편의점 많이 쓰는데 할인 카드 추천해줘")
        # 1. Tool 선택: get_card_recommendation (구어체로 판단)
        # 2. Tool 내부 전처리: "편의점 할인 카드 추천" (검색 최적화)
        # 3. MCP 서버 호출
    """
    
    def __init__(self, tools: List[BaseTool]):
        """
        QueryRouterAgent 초기화
        
        Args:
            tools: 사용할 Tool 리스트 (MCP Tools)
        """
        self._llm = tool_selector_llm  # Tool 선택용 LLM
        self._tools = {tool.name: tool for tool in tools}
        self._llm_with_tools = tool_selector_llm.bind_tools(tools)  # Tool 바인딩
        
        print(f"\n{'='*60}")
        print(f"[QueryRouterAgent 초기화]")
        print(f"Tool 선택 모델: {settings.ROUTING_MODEL_NAME or settings.OLLAMA_MODEL_NAME}")
        print(f"등록된 Tools: {list(self._tools.keys())}")
        print(f"{'='*60}\n")
    
    def run(self, query: str) -> str:
        """
        [동기식 실행]
        사용자 쿼리를 받아 LLM이 자동으로 Tool을 선택하고 실행합니다.
        
        Args:
            query: 사용자의 원본 쿼리
            
        Returns:
            str: Tool 실행 결과 또는 LLM 직접 답변
        """
        
        print(f"\n{'#'*60}")
        print(f"[QueryRouterAgent 실행 시작]")
        print(f"쿼리: {query}")
        print(f"{'#'*60}")
        
        # 시스템 프롬프트: Tool 선택 지침
        system_message = (
            "당신은 사용자 요청을 분석하여 적절한 도구를 선택하는 어시스턴트입니다.\n\n"
            "도구 사용 기준:\n"
            "- 'get_card_recommendation': 카드 추천 요청 (할인 카드, 쇼핑 카드, 편의점 카드 등)\n"
            "- 'analyze_consumption_pattern': 소비 패턴 분석 요청 (통계, 어디에 돈 쓰나 등)\n"
            "- 'query_faq_database': 간단한 정보 조회 (이벤트, FAQ, 신청 방법 등)\n"
            "- 도구 없이 직접 답변: 일반 대화, 인사, 간단한 질문\n\n"
            "사용자의 의도를 정확히 파악하여 가장 적합한 도구를 선택하거나 직접 답변하세요."
        )
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=query)
        ]
        
        try:
            print(f"--- [Agent] 1. LLM 호출 (Tool 선택) ---")
            response = self._llm_with_tools.invoke(messages)
            
            # Tool 호출 여부 확인
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                print(f"--- [Agent] 2. Tool 선택됨: {tool_name} ---")
                print(f"--- [Agent] 인자: {tool_args} ---")
                
                if tool_name in self._tools:
                    # Tool 실행 - query 인자만 전달
                    query_input = tool_args.get("query", "")
                    result = self._tools[tool_name].run(query_input)
                    
                    print(f"--- [Agent] 3. Tool 실행 완료 ---")
                    print(f"{'#'*60}\n")
                    
                    return result
                else:
                    print(f"--- [Agent] ERROR: 알 수 없는 Tool '{tool_name}' ---")
                    return f"[오류: '{tool_name}' 도구를 찾을 수 없습니다.]"
            
            # Tool 없이 직접 답변
            print(f"--- [Agent] 2. LLM 직접 답변 ---")
            direct_answer = str(response.content)
            
            print(f"--- [Agent] 3. 답변 완료 ---")
            print(f"{'#'*60}\n")
            
            return direct_answer
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"[QueryRouterAgent 오류]")
            print(f"오류: {e}")
            print(f"{'!'*60}\n")
            
            return f"[에이전트 실행 오류: {e}]"


# --- Tool 인스턴스 생성 ---
card_tool = MCPCardRecommendationTool()
mydata_tool = MCPMyDataAnalysisTool()
qna_tool = MCPQnADatabaseTool()

# --- 에이전트 인스턴스 생성 ---
query_router_agent = QueryRouterAgent(tools=[card_tool, mydata_tool, qna_tool])
