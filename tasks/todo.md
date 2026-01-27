# LLM Serving Benchmark - Todo

## Problem
vLLM vs TGI vs transformers 서빙 프레임워크 성능 비교

## Plan

### To-Do List
- [x] benchmark_vllm.py 구현
- [x] benchmark_tgi.py 구현
- [x] benchmark_transformers.py 구현
- [x] load_test.py 구현 (동시 요청 부하 테스트)
- [x] analyze_results.py 구현 (결과 분석 및 시각화)
- [x] docker-compose.yaml 작성
- [x] results/ 템플릿 파일 작성
- [x] requirements.txt 작성
- [x] README.md 작성
- [x] 커밋 및 푸시

## Review

### 구현 완료 내용

1. **benchmark_vllm.py**
   - vLLM OpenAI 호환 API 벤치마크
   - 단일 요청 지연 시간 측정
   - 처리량 테스트

2. **benchmark_tgi.py**
   - TGI generate API 벤치마크
   - OpenAI 호환 엔드포인트 지원

3. **benchmark_transformers.py**
   - 직접 추론 벤치마크 (베이스라인)
   - GPU 메모리 사용량 측정

4. **load_test.py**
   - asyncio + aiohttp 기반 동시 요청
   - 10, 50, 100 동시 요청 테스트
   - RPS, P50/P90/P99 지연 시간 측정

5. **analyze_results.py**
   - matplotlib 기반 시각화
   - 프레임워크 비교 차트 생성
   - 마크다운 요약 생성

### 사용 방법

```bash
# 서비스 시작
docker-compose up -d

# 벤치마크 실행
python benchmark_vllm.py
python benchmark_tgi.py
python load_test.py

# 결과 분석
python analyze_results.py
```
