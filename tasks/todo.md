# LLM Quantization Pipeline - Todo

## Problem
오픈소스 LLM을 GPTQ, GGUF로 양자화하고 성능을 비교하는 파이프라인 구축

## Plan

### To-Do List
- [x] Git 저장소 초기화 및 사용자 설정
- [x] requirements.txt 작성
- [x] quantize_gptq.py 구현 (AutoGPTQ 4bit 양자화)
- [x] convert_gguf.py 구현 (GGUF 변환)
- [x] inference_gptq.py 구현 (GPTQ 추론)
- [x] inference_gguf.py 구현 (GGUF 추론)
- [x] benchmark.py 구현 (성능 비교)
- [x] results/benchmark_results.md 작성
- [x] README.md 작성
- [x] 커밋 및 푸시

## 프로젝트 구조
```
llm-quantization-pipeline/
├── quantize_gptq.py      # GPTQ 양자화
├── convert_gguf.py       # GGUF 변환
├── benchmark.py          # 성능 비교
├── inference_gptq.py     # GPTQ 추론
├── inference_gguf.py     # GGUF 추론
├── results/
│   └── benchmark_results.md
├── requirements.txt
├── README.md
└── tasks/
    └── todo.md
```

## Review

### 구현 완료 내용

1. **quantize_gptq.py**
   - AutoGPTQ를 사용한 4-bit 양자화
   - 캘리브레이션 데이터 자동 생성
   - 커맨드라인 인터페이스 제공

2. **convert_gguf.py**
   - llama.cpp를 이용한 GGUF 변환
   - Q4_K_M, Q5_K_M 양자화 지원
   - HuggingFace 모델 자동 다운로드

3. **inference_gptq.py / inference_gguf.py**
   - 단일 프롬프트 및 대화 모드 지원
   - 생성 속도 측정 (tokens/sec)

4. **benchmark.py**
   - 원본/GPTQ/GGUF 모델 성능 비교
   - 메모리 사용량, 속도, 품질 측정
   - 마크다운 리포트 자동 생성

### 사용 방법

```bash
# GPTQ 양자화
python quantize_gptq.py --model Qwen/Qwen2.5-3B-Instruct

# GGUF 변환
python convert_gguf.py --model Qwen/Qwen2.5-3B-Instruct

# 벤치마크
python benchmark.py --gptq ./models/gptq-4bit --gguf ./models/gguf/model.gguf
```
