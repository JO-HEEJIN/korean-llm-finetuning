# LLM Quantization Benchmark Results

## Summary

| Model Type | Quantization | Memory (MB) | Speed (tokens/s) | Quality |
|------------|--------------|-------------|------------------|---------|
| Original   | FP16         | ~6000       | ~30              | Baseline |
| GPTQ       | 4-bit        | ~2000       | ~50              | Good |
| GGUF       | Q4_K_M       | ~2200       | ~45              | Good |
| GGUF       | Q5_K_M       | ~2600       | ~40              | Better |

*Note: These are estimated values. Run benchmark.py for actual results.*

## Test Environment

- GPU: (your GPU here)
- VRAM: (your VRAM here)
- Model: Qwen2.5-3B-Instruct

## How to Run Benchmark

```bash
# Benchmark all models
python benchmark.py \
    --original Qwen/Qwen2.5-3B-Instruct \
    --gptq ./models/gptq-4bit \
    --gguf ./models/gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# Benchmark specific model
python benchmark.py --gptq ./models/gptq-4bit
```

## Observations

### Memory Usage
- GPTQ 4-bit reduces memory by approximately 60-70%
- GGUF Q4_K_M provides similar memory savings

### Inference Speed
- GPTQ typically offers faster inference on GPU
- GGUF is efficient on both CPU and GPU

### Quality
- 4-bit quantization generally maintains good quality
- Q5_K_M offers slightly better quality than Q4_K_M
