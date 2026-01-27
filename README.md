# Korean LLM Fine-tuning

한국어 고객 상담 특화 LLM 파인튜닝 프로젝트

## 개요

이 프로젝트는 LoRA (Low-Rank Adaptation)를 사용하여 한국어 고객 상담에 특화된 LLM을 효율적으로 파인튜닝합니다.

### 주요 특징

- **LoRA 파인튜닝**: 전체 모델 대신 작은 어댑터만 학습하여 메모리 효율적
- **4-bit 양자화**: bitsandbytes를 사용한 4-bit 양자화로 GPU 메모리 절약
- **유연한 설정**: YAML 기반 설정으로 쉽게 하이퍼파라미터 조정 가능

## 프로젝트 구조

```
korean-llm-finetuning/
├── train.py              # LoRA 파인튜닝 스크립트
├── inference.py          # 추론 및 대화 모드
├── dataset.py            # 데이터셋 전처리 모듈
├── config.yaml           # 설정 파일
├── data/
│   └── sample_instructions.jsonl  # 샘플 학습 데이터
├── requirements.txt      # 의존성 목록
└── README.md
```

## 설치

### 1. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. GPU 요구사항

- CUDA 지원 GPU (최소 8GB VRAM 권장)
- CUDA 11.8 이상

## 사용법

### 학습 (Fine-tuning)

```bash
# 기본 설정으로 학습
python train.py --config config.yaml

# 테스트용 짧은 학습 (1 step만)
python train.py --config config.yaml --max_steps 1
```

### 추론 (Inference)

```bash
# 대화 모드 (베이스 모델)
python inference.py --config config.yaml

# 파인튜닝된 모델로 대화
python inference.py --config config.yaml --adapter_path ./outputs/final_model

# 단일 프롬프트 테스트
python inference.py --config config.yaml --prompt "배송이 언제 도착하나요?"
```

## 설정

`config.yaml`에서 다음 설정을 조정할 수 있습니다:

### 모델 설정

```yaml
model:
  base_model: "Qwen/Qwen2.5-3B"  # 또는 "meta-llama/Llama-3.2-3B"
```

### LoRA 하이퍼파라미터

```yaml
lora:
  r: 16                # LoRA rank
  lora_alpha: 32       # LoRA alpha
  lora_dropout: 0.05   # Dropout 비율
  target_modules:      # 적용할 모듈
    - q_proj
    - v_proj
    - k_proj
    - o_proj
```

### 학습 설정

```yaml
training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

## 데이터 형식

학습 데이터는 JSONL 형식으로 제공해야 합니다:

```json
{"instruction": "질문 또는 요청", "output": "응답"}
```

예시:
```json
{"instruction": "배송이 언제 도착하나요?", "output": "안녕하세요, 고객님. 주문하신 상품의 배송 조회를 도와드리겠습니다..."}
```

## 커스텀 데이터로 학습

1. `data/` 폴더에 JSONL 형식의 데이터 파일 추가
2. `config.yaml`의 `data.train_file` 경로 수정
3. `python train.py` 실행

### HuggingFace 데이터셋 사용

`config.yaml`에서 다음과 같이 설정:

```yaml
data:
  huggingface_dataset: "beomi/KoAlpaca-v1.1a"
```

## 결과 비교

### 파인튜닝 전 (베이스 모델)

```
고객: 배송이 언제 도착하나요?
AI: [일반적이고 맥락에 맞지 않는 응답]
```

### 파인튜닝 후

```
고객: 배송이 언제 도착하나요?
AI: 안녕하세요, 고객님. 주문하신 상품의 배송 조회를 도와드리겠습니다.
    주문번호를 알려주시면 정확한 배송 현황을 확인해 드릴 수 있습니다.
```

## 문제 해결

### CUDA Out of Memory

- `config.yaml`에서 `per_device_train_batch_size` 감소
- `gradient_accumulation_steps` 증가
- `max_seq_length` 감소

### bitsandbytes 설치 오류

```bash
pip install bitsandbytes --prefer-binary
```

## 라이선스

MIT License

## 참고 자료

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
