# On-Premise LLM Server - Todo

## Problem
사내망/로컬 환경에서 LLM을 서빙하는 프로덕션 레벨 시스템 구축

## Plan

### To-Do List
- [x] config.py 작성 (설정 관리)
- [x] api_server.py 구현 (FastAPI + 인증 + 로깅 + 레이트 리밋)
- [x] health_check.py 구현 (헬스체크)
- [x] client.py 구현 (API 클라이언트 예제)
- [x] Dockerfile 작성
- [x] docker-compose.yaml 작성
- [x] nginx/nginx.conf 작성
- [x] requirements.txt 작성
- [x] README.md 작성
- [x] 커밋 및 푸시

## 프로젝트 구조
```
onprem-llm-server/
├── docker-compose.yaml
├── Dockerfile
├── api_server.py
├── client.py
├── config.py
├── health_check.py
├── nginx/
│   └── nginx.conf
├── requirements.txt
├── README.md
└── tasks/
    └── todo.md
```

## Review

### 구현 완료 내용

1. **config.py**
   - pydantic-settings 기반 설정 관리
   - 환경 변수 및 .env 파일 지원
   - 모델별 프리셋 설정

2. **api_server.py**
   - OpenAI 호환 REST API (/v1/chat/completions, /v1/completions)
   - Bearer 토큰 인증
   - slowapi 기반 레이트 리밋
   - Prometheus 메트릭 수집
   - 요청 로깅

3. **docker-compose.yaml**
   - vLLM 서비스 (GPU 지원)
   - API 서버 서비스
   - Nginx 리버스 프록시
   - 헬스체크 설정

4. **nginx/nginx.conf**
   - 리버스 프록시 설정
   - 레이트 리밋
   - 보안 헤더
   - 스트리밍 지원

### 사용 방법

```bash
# 서비스 시작
docker-compose up -d

# API 테스트
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```
