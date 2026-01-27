# On-Premise LLM Server

사내망/로컬 환경에서 LLM을 서빙하는 프로덕션 레벨 시스템

## Overview

vLLM을 백엔드로 사용하여 OpenAI 호환 API를 제공합니다.

### Features

- OpenAI 호환 REST API
- 인증 (API Key)
- 레이트 리밋
- 요청 로깅
- Prometheus 메트릭
- 헬스체크
- Nginx 리버스 프록시

## Project Structure

```
onprem-llm-server/
├── docker-compose.yaml   # 컨테이너 구성
├── Dockerfile            # API 서버 이미지
├── api_server.py         # FastAPI 서버
├── client.py             # API 클라이언트 예제
├── config.py             # 설정 관리
├── health_check.py       # 헬스체크 유틸리티
├── nginx/
│   └── nginx.conf        # Nginx 설정
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# .env 파일 생성
cat > .env << EOF
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
ENABLE_AUTH=false
API_KEY=your-secret-key
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
EOF
```

### 2. Start Services

```bash
# 전체 스택 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. Test API

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Qwen/Qwen2.5-3B-Instruct | HuggingFace model name |
| `MODEL_PATH` | ./models | Local model path |
| `TENSOR_PARALLEL_SIZE` | 1 | GPU parallelism |
| `GPU_MEMORY_UTILIZATION` | 0.9 | GPU memory usage ratio |
| `MAX_MODEL_LEN` | 4096 | Maximum sequence length |
| `ENABLE_AUTH` | false | Enable API authentication |
| `API_KEY` | - | API key for authentication |
| `RATE_LIMIT_REQUESTS` | 60 | Requests per minute |
| `LOG_LEVEL` | INFO | Logging level |

## Client Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Performance Tuning

### GPU Memory

```bash
# 더 많은 GPU 메모리 사용
GPU_MEMORY_UTILIZATION=0.95 docker-compose up -d
```

### Multiple GPUs

```bash
# 2개 GPU 사용
TENSOR_PARALLEL_SIZE=2 docker-compose up -d
```

### Batch Size

vLLM은 자동으로 요청을 배치 처리합니다. `MAX_MODEL_LEN`을 조정하여 처리량을 최적화할 수 있습니다.

## Monitoring

### Prometheus Metrics

- `llm_requests_total`: 총 요청 수
- `llm_request_latency_seconds`: 요청 지연 시간
- `llm_tokens_total`: 총 토큰 수

### Health Check

```bash
# API 서버 상태
curl http://localhost:8000/health

# vLLM 백엔드 상태
curl http://localhost:8001/health
```

## Troubleshooting

### GPU not detected

```bash
# NVIDIA driver 확인
nvidia-smi

# Docker GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Out of Memory

- `GPU_MEMORY_UTILIZATION` 값 낮추기
- `MAX_MODEL_LEN` 값 줄이기
- 더 작은 모델 사용

### Slow Inference

- `TENSOR_PARALLEL_SIZE` 증가 (멀티 GPU)
- 모델 양자화 버전 사용

## License

MIT License
