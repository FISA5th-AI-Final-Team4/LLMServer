"""
쿼리 라우터 에이전트 (LangChain Tool 기반)
- 1단계: 쿼리 전처리 (Query Preprocessing)
- 2단계: LangChain Tool 자동 선택 → MCP 서버 호출 (operation_id 매칭)
"""

from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

import json
import httpx
import asyncio
from typing import Dict, Any, List

from core.config import settings
from schemas.mcp_router import PreprocessedQuery, RouteType, RoutingDecision


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
# 1단계: 쿼리 전처리 (Query Preprocessing)
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
# 2단계: MCP Tools 정의 (operation_id 매칭)
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
        
        # 1단계: 쿼리 전처리
        print(f"--- [CardTool] 1. 쿼리 전처리 시작 ---")
        preprocessed = _preprocess_query_internal(query)
        
        # 2단계: MCP 서버 호출
        base_url = MCP_URL.replace('/mcp', '')
        endpoint_url = f"{base_url}/tools/card-recommendation"
        
        print(f"--- [CardTool] 2. MCP 서버 호출: {endpoint_url} ---")
        print(f"--- [CardTool] 정제된 쿼리: {preprocessed.normalized_query} ---")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    endpoint_url,
                    json={
                        "query": preprocessed.normalized_query,
                        "retrieve_k": 5,
                        "final_k": 3
                    },
                    timeout=60.0
                )
                
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
                
                print(f"--- [CardTool] 3. 응답 완료 ---")
                return formatted_response
                
            except Exception as e:
                print(f"--- [CardTool] ERROR: {e} ---")
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


# =====================================================
# 기존 라우팅 함수 (폐기 예정 - 호환성 유지용)
# =====================================================

def route_query(preprocessed_query: PreprocessedQuery) -> RoutingDecision:
    """
    전처리된 쿼리를 분석하여 적절한 도구로 라우팅합니다.
    
    라우팅 경로:
    - RAG_SEARCH: 카드 추천, 금융 용어 설명 등 RAG 검색 필요
    - ML_TOOL: 소비 패턴 분석, 예측 등 ML 모델 필요
    - QNA_DB: 간단한 정보 조회 (이벤트, FAQ 등)
    - GENERAL: 일반 대화, 인사 등
    
    Args:
        preprocessed_query: 전처리된 쿼리 객체
        
    Returns:
        RoutingDecision: 라우팅 결정 결과 (경로, 이유, 신뢰도)
    """
    
    print(f"\n{'='*60}")
    print(f"[2단계: 쿼리 라우팅 시작]")
    print(f"정제된 쿼리: {preprocessed_query.normalized_query}")
    print(f"키워드: {', '.join(preprocessed_query.key_keywords)}")
    print(f"{'='*60}")
    
    # 시스템 프롬프트: 라우팅 지침
    system_prompt = """당신은 사용자 쿼리를 분석하여 적절한 처리 경로로 라우팅하는 전문가입니다.

**4가지 라우팅 경로**:

1. **RAG_SEARCH** - RAG 검색 (벡터 DB 검색 + LLM 생성)
   - 카드 추천: "할인 카드 추천", "쇼핑 카드 알려줘", "주유 카드 뭐가 좋아?"
   - 금융 용어 설명: "연회비가 뭐야?", "APR이란?", "체크카드 신용카드 차이"
   - 카드 혜택 정보: "이 카드 혜택 알려줘", "할인율 얼마야?"

2. **ML_TOOL** - ML 모델 (데이터 분석, 예측)
   - 소비 패턴 분석: "내 소비 패턴 분석해줘", "어디에 돈 많이 쓰나?"
   - 추천 예측: "나한테 맞는 카드 예측해줘" (MyData 기반)
   - 통계 분석: "이번 달 소비 통계"

3. **QNA_DB** - 간단한 정보 조회 (DB 쿼리)
   - 이벤트 정보: "이벤트 언제까지?", "프로모션 있어?"
   - FAQ: "카드 신청 방법", "고객센터 번호"
   - 간단한 사실 확인: "영업시간", "수수료 얼마?"

4. **GENERAL** - 일반 대화
   - 인사: "안녕", "고마워", "도와줘"
   - 잡담: "날씨 좋네", "오늘 뭐하지?"
   - 기타 대화

**응답 형식** (반드시 JSON만 출력):
{
  "route": "RAG_SEARCH" | "ML_TOOL" | "QNA_DB" | "GENERAL",
  "reason": "선택 이유를 한 문장으로",
  "confidence": 0.0 ~ 1.0
}

**예시**:

입력: "20대 여성 쇼핑 할인 카드 추천"
→ {"route": "RAG_SEARCH", "reason": "카드 추천 요청으로 RAG 검색 필요", "confidence": 0.95}

입력: "내 소비 패턴 분석해줘"
→ {"route": "ML_TOOL", "reason": "소비 패턴 분석은 ML 모델 필요", "confidence": 0.9}

입력: "이벤트 언제까지?"
→ {"route": "QNA_DB", "reason": "이벤트 정보는 DB 조회로 충분", "confidence": 0.85}

입력: "안녕하세요"
→ {"route": "GENERAL", "reason": "일반 인사말", "confidence": 1.0}

**중요**: 
- 응답은 반드시 JSON만 (다른 텍스트 금지)
- route는 반드시 위 4가지 중 하나
- confidence는 결정에 대한 확신도"""

    # 전처리된 쿼리 정보를 사용자 프롬프트에 포함
    user_prompt = f"""다음 쿼리를 분석하여 적절한 경로로 라우팅하세요:

**정제된 쿼리**: {preprocessed_query.normalized_query}
**핵심 키워드**: {', '.join(preprocessed_query.key_keywords)}
**원본 쿼리**: {preprocessed_query.original_query}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        # LLM 호출
        print(f"[라우팅 LLM 호출 중...]")
        response = routing_llm.invoke(messages)
        raw_output = response.content
        
        print(f"[LLM 원본 응답]\n{raw_output}\n")
        
        # JSON 파싱
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            json_str = raw_output.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_output.strip()
        
        parsed_data = json.loads(json_str)
        
        # RouteType enum으로 변환
        route_str = parsed_data.get("route", "GENERAL")
        try:
            route = RouteType[route_str]
        except KeyError:
            print(f"[경고] 알 수 없는 경로: {route_str}, GENERAL로 폴백")
            route = RouteType.GENERAL
        
        # RoutingDecision 객체 생성
        result = RoutingDecision(
            route=route,
            reason=parsed_data.get("reason", "라우팅 결정"),
            confidence=float(parsed_data.get("confidence", 0.5)),
            preprocessed_query=preprocessed_query
        )
        
        print(f"[라우팅 완료]")
        print(f"✓ 선택된 경로: {result.route.value}")
        print(f"✓ 선택 이유: {result.reason}")
        print(f"✓ 신뢰도: {result.confidence:.2f}")
        print(f"{'='*60}\n")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파싱 실패: {e}")
        print(f"[폴백] GENERAL 경로 사용")
        
        # 파싱 실패 시 GENERAL로 폴백
        return RoutingDecision(
            route=RouteType.GENERAL,
            reason="라우팅 파싱 실패로 일반 대화 처리",
            confidence=0.3,
            preprocessed_query=preprocessed_query
        )
        
    except Exception as e:
        print(f"[오류] 라우팅 실패: {e}")
        print(f"[폴백] GENERAL 경로 사용")
        
        return RoutingDecision(
            route=RouteType.GENERAL,
            reason="라우팅 오류로 일반 대화 처리",
            confidence=0.0,
            preprocessed_query=preprocessed_query
        )


# =====================================================
# 통합 에이전트 (weather_agent.AgentRunner 스타일)
# =====================================================

class QueryRouterAgent:
    """
    LangChain Tool 기반 쿼리 라우터 에이전트
    
    동작 방식:
    1. 사용자 쿼리 입력
    2. LLM이 적절한 Tool 자동 선택 (get_card_recommendation, analyze_consumption_pattern 등)
    3. 선택된 Tool 내부에서 쿼리 전처리 수행
    4. MCP 서버 호출 (operation_id 자동 매칭)
    5. 결과 반환
    
    사용 예시:
        agent = QueryRouterAgent()
        result = agent.run("편의점 많이 쓰는데 할인 카드 추천해줘")
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
                    # Tool 실행
                    result = self._tools[tool_name].run(**tool_args)
                    
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
