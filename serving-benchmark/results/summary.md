# LLM Serving Benchmark Results

## Framework Comparison

| Framework | Avg Latency (ms) | P99 Latency (ms) | Tokens/sec |
|-----------|------------------|------------------|------------|
| vLLM | - | - | - |
| TGI | - | - | - |
| Transformers | - | - | - |

*Run benchmarks to populate this table*

## Load Test Results

| Framework | Concurrency | RPS | P99 Latency (ms) |
|-----------|-------------|-----|------------------|
| vLLM | 10 | - | - |
| vLLM | 50 | - | - |
| vLLM | 100 | - | - |
| TGI | 10 | - | - |
| TGI | 50 | - | - |
| TGI | 100 | - | - |

## Expected Results

Based on typical benchmarks:

### vLLM
- Best throughput at high concurrency
- Optimized for continuous batching
- Lower memory overhead per request

### TGI
- Good balance of latency and throughput
- Simpler deployment
- Native streaming support

### Transformers
- Baseline performance
- No serving optimizations
- Highest memory usage

## How to Run

```bash
# Start services
docker-compose up -d

# Wait for models to load
sleep 120

# Run benchmarks
python benchmark_vllm.py --requests 100
python benchmark_tgi.py --requests 100
python benchmark_transformers.py --requests 50

# Run load tests
python load_test.py --concurrency 10 50 100

# Analyze results
python analyze_results.py
```
