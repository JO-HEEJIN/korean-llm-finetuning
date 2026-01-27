# LLM Serving Benchmark

vLLM vs TGI vs Transformers 성능 비교 벤치마크

## Overview

세 가지 LLM 서빙 방식의 성능을 비교합니다:
- **vLLM**: 고성능 LLM 서빙 엔진
- **TGI**: HuggingFace Text Generation Inference
- **Transformers**: 직접 추론 (베이스라인)

## Project Structure

```
llm-serving-benchmark/
├── benchmark_vllm.py         # vLLM 벤치마크
├── benchmark_tgi.py          # TGI 벤치마크
├── benchmark_transformers.py # Transformers 벤치마크
├── load_test.py              # 동시 요청 부하 테스트
├── analyze_results.py        # 결과 분석 및 시각화
├── docker-compose.yaml       # vLLM, TGI 컨테이너
├── results/
│   ├── benchmark_comparison.png
│   ├── detailed_metrics.json
│   └── summary.md
├── requirements.txt
└── README.md
```

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| Latency | 요청당 응답 시간 (ms) |
| Throughput | 초당 생성 토큰 수 (tokens/sec) |
| RPS | 초당 처리 요청 수 |
| P50/P90/P99 | 백분위 지연 시간 |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Serving Frameworks

```bash
# vLLM + TGI 동시 실행 (2개 GPU 필요)
docker-compose up -d

# 모델 로딩 대기 (약 2분)
sleep 120

# 상태 확인
curl http://localhost:8001/health  # vLLM
curl http://localhost:8002/health  # TGI
```

### 3. Run Benchmarks

```bash
# 개별 벤치마크
python benchmark_vllm.py --requests 100
python benchmark_tgi.py --requests 100
python benchmark_transformers.py --requests 50

# 부하 테스트
python load_test.py --concurrency 10 50 100 --requests 100

# 결과 분석 및 시각화
python analyze_results.py
```

## Benchmark Scenarios

### 1. Single Request Latency
다양한 입력/출력 길이에서 단일 요청 지연 시간 측정

### 2. Throughput
배치 크기별 초당 처리량 측정

### 3. Concurrent Requests
10, 50, 100 동시 요청 처리 성능

### 4. Memory Usage
GPU 메모리 사용량 비교

## Expected Results

| Framework | Avg Latency | Throughput | Concurrent Handling |
|-----------|-------------|------------|---------------------|
| vLLM | Low | High | Excellent |
| TGI | Medium | Medium-High | Good |
| Transformers | High | Low | Poor |

## Configuration

### Environment Variables

```bash
# Model selection
export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

# GPU allocation
export CUDA_VISIBLE_DEVICES=0,1
```

### docker-compose.yaml

- `vllm`: GPU 0, Port 8001
- `tgi`: GPU 1, Port 8002

## Output Files

### results/benchmark_comparison.png
성능 비교 차트:
- Latency comparison
- Throughput comparison
- Load test results

### results/detailed_metrics.json
상세 측정 데이터 (JSON 형식)

### results/summary.md
마크다운 형식 결과 요약

## Single GPU Setup

GPU가 1개인 경우:

```bash
# vLLM만 실행
docker-compose up vllm -d

# 또는 TGI만 실행
docker-compose up tgi -d
```

## Troubleshooting

### Out of Memory
- `--gpu-memory-utilization` 값 낮추기
- `--max-model-len` 값 줄이기

### Slow Model Loading
- HuggingFace 캐시 확인
- 네트워크 상태 확인

### Connection Refused
- 서비스 시작 후 충분한 대기 시간 필요
- 헬스체크 엔드포인트 확인

## License

MIT License
