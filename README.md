# LLM Quantization Pipeline

LLM 양자화 및 성능 비교 파이프라인

## Overview

오픈소스 LLM을 GPTQ, GGUF 형식으로 양자화하고 성능을 비교합니다.

### Supported Quantization Methods

| Method | Format | Bits | Use Case |
|--------|--------|------|----------|
| GPTQ   | Safetensors | 4-bit | GPU inference |
| GGUF   | Q4_K_M | 4-bit | CPU/GPU inference |
| GGUF   | Q5_K_M | 5-bit | Higher quality |

## Project Structure

```
llm-quantization-pipeline/
├── quantize_gptq.py      # GPTQ quantization
├── convert_gguf.py       # GGUF conversion
├── benchmark.py          # Performance comparison
├── inference_gptq.py     # GPTQ inference
├── inference_gguf.py     # GGUF inference
├── results/
│   └── benchmark_results.md
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GGUF conversion, clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
export LLAMA_CPP_PATH=/path/to/llama.cpp
```

## Usage

### 1. GPTQ Quantization

```bash
# Quantize model to 4-bit GPTQ
python quantize_gptq.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output ./models/gptq-4bit \
    --bits 4
```

### 2. GGUF Conversion

```bash
# Convert to GGUF format
python convert_gguf.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output ./models/gguf \
    --quant-types Q4_K_M Q5_K_M
```

### 3. Inference

```bash
# GPTQ inference
python inference_gptq.py --model ./models/gptq-4bit

# GGUF inference
python inference_gguf.py --model ./models/gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# Single prompt
python inference_gptq.py --model ./models/gptq-4bit --prompt "Hello, how are you?"
```

### 4. Benchmark

```bash
# Compare all models
python benchmark.py \
    --original Qwen/Qwen2.5-3B-Instruct \
    --gptq ./models/gptq-4bit \
    --gguf ./models/gguf/model-Q4_K_M.gguf \
    --output ./results/benchmark_results.md
```

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| Memory (MB) | GPU/RAM usage during inference |
| Speed (tokens/s) | Token generation speed |
| Quality | Response quality comparison |

## Expected Results

| Model | Memory | Speed | Notes |
|-------|--------|-------|-------|
| Original (FP16) | ~6GB | ~30 t/s | Baseline |
| GPTQ 4-bit | ~2GB | ~50 t/s | Best for GPU |
| GGUF Q4_K_M | ~2.2GB | ~45 t/s | CPU/GPU flexible |
| GGUF Q5_K_M | ~2.6GB | ~40 t/s | Better quality |

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU inference)
- 8GB+ VRAM recommended

## Troubleshooting

### GPTQ Issues
- Ensure CUDA is properly installed
- Check GPU memory availability

### GGUF Issues
- Set LLAMA_CPP_PATH environment variable
- Build llama.cpp with CUDA support for GPU acceleration

## License

MIT License
