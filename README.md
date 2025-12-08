# 챗봇의 정석 LLM Server

| 소개 | 관련 링크 |
| ------ | ------ |
| 챗봇의 정석의 **두뇌 역할을 하는 LLM 서버 원격 저장소**입니다.<br>qwen3:1.7b 모델을 Ollama로 호스팅하며, 사용자의 채팅을 입력 받아<br> **LangGraph 기반 커스텀 에이전트**를 통해 MCP 툴을 자동으로 선택·실행하고, 그 결과를 백엔드 서버로 전달합니다.<br>MCP 툴이 필요 없는 일반 대화의 경우에는, 도구를 호출하지 않고 **LLM의 기본 응답**만 반환합니다. | 🔗[챗봇의 정석](https://github.com/FISA5th-AI-Final-Team4) <br>🔗[FrontEnd](https://github.com/FISA5th-AI-Final-Team4/FrontEnd) <br>🔗[BackEnd](https://github.com/FISA5th-AI-Final-Team4/BackEnd) <br>🔗[MCP서버](https://github.com/FISA5th-AI-Final-Team4/MCPServer) <br>🔗[DB서버](https://github.com/FISA5th-AI-Final-Team4/LocalDbSetup) |


## 🤖 LangGraph 에이전트 아키텍처
- 경량 LLM 모델인 qwen3:1.7b를 Ollama로 호스팅하여 사용자 응답 속도를 높이고, MCP 툴 응답을 전달해 정확한 정보에 기반한 답변을 제공합니다.
- LangGraph 기반 커스텀 에이전트를 설계하여, 정확한 UUID가 필요한 MCP 툴 호출 시 백엔드 서버에서 전달받은 세션 ID를 호출 인자에 자동 주입함으로써 툴 호출 오류를 차단했습니다.
<img height="480" alt="image" src="https://github.com/user-attachments/assets/8d85e832-f533-45e0-8a5a-68f07554e909" />

## 🧾 엔드포인트
- `POST /llm/mcp-router/dispatch`: 시스템 프롬프트 + 사용자 쿼리를 LangGraph 에이전트에 전달해 답변 및 `tool_response` 반환
- `POST /llm/mcp-router/echo`: 연결 확인용 에코

## ⚒️ 기술 스택
- FastAPI, pydantic

  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white"> 

- LangGraph, langchain-core, Ollama

  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=LangGraph&logoColor=white"/> <img src="https://img.shields.io/badge/langchain-core-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/> <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white"/>

- Docker + uvicorn + EC2

    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white"> <img src="https://img.shields.io/badge/uvicorn-E4CCFF?style=for-the-badge"> <img src="https://img.shields.io/badge/AWS EC2-F58536?style=for-the-badge"/>


## 📁 프로젝트 구조
```bash
.
├── Dockerfile                     # 컨테이너 이미지를 빌드하기 위한 Docker 설정
├── api                            # FastAPI 라우터 및 엔드포인트 정의
│   ├── router.py                  # 공통 APIRouter 설정 및 라우터 등록
│   └── routes                     # 실제 HTTP 엔드포인트 모듈
│       └── mcp_router.py          # MCP Router 엔드포인트 (Back → LLM → MCP 툴 디스패치 API)
├── core                           # 핵심 로직 및 공통 유틸 모듈
│   ├── config.py                  # 환경변수, 설정값 로딩 및 글로벌 Settings 정의
│   ├── dep.py                     # LangGraph 에이전트/툴 의존성 주입 함수
│   ├── model_config.py            # LLM 시스템 프롬프트 정의
│   ├── parse_tool.py              # MCP 툴 응답 파싱 및 변환 로직
│   ├── setup.py                   # 앱 초기화 AI Agent 생성 및 MCP 서버 연결 설정
│   └── trace_agent.py             # LLM 호출/툴 실행에 대한 트레이싱 및 로깅/모니터링 로직
├── docker-compose.cpu.yml         # CPU 환경용 Docker Compose 설정
├── docker-compose.gpu.yml         # GPU 환경용 Docker Compose 설정
├── main.py                        # FastAPI 앱 엔트리포인트 (uvicorn 실행 대상)
├── requirements.in                # 의존성 원본 목록 (pip-compile 입력용)
├── requirements.txt               # 고정된 의존성 버전 목록 (배포/빌드용)
└── schemas                        # Pydantic 스키마 정의
    └── mcp_router.py              # 디스패치 요청/응답 바디 스키마
```

## ⚙️ 환경 변수 & 서버 실행
- 환경 변수 (`.env`)
    ```bash
    BACKEND_HOST=http://127.0.0.1:8001      # CORS 허용 도메인/포트
    OLLAMA_MODEL_NAME=qwen3:1.7b            # OLLAMA 모델 이름 (LLM Agent)
    OLLAMA_BASE_URL=http://127.0.0.1:11434  # OLLAMA 서버 주소
    MCP_SERVER_URL=http://127.0.0.1:8011    # MCP 서버 주소

    # LangSmith 추적 설정
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=https://api.langchain.com
    LANGCHAIN_API_KEY=your_api_key
    LANGCHAIN_PROJECT=your_project
    LANGCHAIN_SESSION=your_session
    ```


- 서버 실행 
    ```bash
    # Docker 이미지 빌드 및 컨테이너 실행 (CPU 사용)
    docker compose --env-file .env -f docker-compose.cpu.yml up --build -d 
    # Docker 이미지 빌드 및 컨테이너 실행 (GPU 사용)
    docker compose --env-file .env -f docker-compose.gpu.yml up --build -d 
    ```
    - 브라우저에서 `http://127.0.0.1:8000/docs` 로 OpenAPI 문서를 확인할 수 있습니다.
