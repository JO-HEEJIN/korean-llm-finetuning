"""
Benchmark Script
Compare performance of original vs quantized models
"""

import argparse
import json
import os
import time
import psutil
import torch
from datetime import datetime

# Optional imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from auto_gptq import AutoGPTQForCausalLM
    HAS_GPTQ = True
except ImportError:
    HAS_GPTQ = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


# Test prompts for benchmarking
TEST_PROMPTS = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about nature.",
    "What are the benefits of exercise?",
    "Describe the process of photosynthesis.",
]


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_reserved():
    """Get reserved GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0


def benchmark_transformers(model_path: str, prompts: list, max_tokens: int = 128):
    """Benchmark standard transformers model."""
    if not HAS_TRANSFORMERS:
        return None

    print(f"\nBenchmarking Transformers model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    load_memory = get_gpu_memory_reserved()

    results = []
    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        end = time.time()

        output_len = outputs.shape[1] - input_len
        gen_time = end - start

        total_tokens += output_len
        total_time += gen_time

        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "response": response[:200],
            "tokens": output_len,
            "time": gen_time,
        })

    return {
        "model_type": "transformers",
        "model_path": model_path,
        "memory_mb": load_memory,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": results,
    }


def benchmark_gptq(model_path: str, prompts: list, max_tokens: int = 128):
    """Benchmark GPTQ model."""
    if not HAS_GPTQ:
        return None

    print(f"\nBenchmarking GPTQ model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )

    load_memory = get_gpu_memory_reserved()

    results = []
    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        end = time.time()

        output_len = outputs.shape[1] - input_len
        gen_time = end - start

        total_tokens += output_len
        total_time += gen_time

        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "response": response[:200],
            "tokens": output_len,
            "time": gen_time,
        })

    return {
        "model_type": "gptq",
        "model_path": model_path,
        "memory_mb": load_memory,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": results,
    }


def benchmark_gguf(model_path: str, prompts: list, max_tokens: int = 128, n_gpu_layers: int = -1):
    """Benchmark GGUF model."""
    if not HAS_LLAMA_CPP:
        return None

    print(f"\nBenchmarking GGUF model: {model_path}")

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    mem_after = process.memory_info().rss / 1024 / 1024
    load_memory = mem_after - mem_before

    results = []
    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        start = time.time()
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
        )
        end = time.time()

        gen_time = end - start
        tokens = output["usage"]["completion_tokens"]

        total_tokens += tokens
        total_time += gen_time

        response = output["choices"][0]["text"].strip()
        results.append({
            "prompt": prompt,
            "response": response[:200],
            "tokens": tokens,
            "time": gen_time,
        })

    return {
        "model_type": "gguf",
        "model_path": model_path,
        "memory_mb": load_memory,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": results,
    }


def generate_report(benchmarks: list, output_path: str):
    """Generate benchmark report in markdown format."""
    report = ["# LLM Quantization Benchmark Results\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary table
    report.append("## Summary\n")
    report.append("| Model Type | Memory (MB) | Speed (tokens/s) |")
    report.append("|------------|-------------|------------------|")

    for bench in benchmarks:
        if bench:
            report.append(
                f"| {bench['model_type']} | {bench['memory_mb']:.1f} | "
                f"{bench['avg_tokens_per_sec']:.1f} |"
            )

    report.append("\n## Detailed Results\n")

    for bench in benchmarks:
        if bench:
            report.append(f"### {bench['model_type'].upper()}\n")
            report.append(f"- Model: `{bench['model_path']}`")
            report.append(f"- Memory Usage: {bench['memory_mb']:.1f} MB")
            report.append(f"- Average Speed: {bench['avg_tokens_per_sec']:.1f} tokens/s")
            report.append(f"- Total Tokens: {bench['total_tokens']}")
            report.append(f"- Total Time: {bench['total_time']:.2f}s\n")

            report.append("#### Sample Outputs\n")
            for i, result in enumerate(bench['results'][:3]):
                report.append(f"**Prompt {i+1}:** {result['prompt']}")
                report.append(f"**Response:** {result['response']}...")
                report.append(f"*({result['tokens']} tokens, {result['time']:.2f}s)*\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized models")
    parser.add_argument("--original", type=str, help="Original model path")
    parser.add_argument("--gptq", type=str, help="GPTQ model path")
    parser.add_argument("--gguf", type=str, help="GGUF model path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--output", type=str, default="./results/benchmark_results.md")

    args = parser.parse_args()

    benchmarks = []

    if args.original:
        result = benchmark_transformers(args.original, TEST_PROMPTS, args.max_tokens)
        benchmarks.append(result)

    if args.gptq:
        result = benchmark_gptq(args.gptq, TEST_PROMPTS, args.max_tokens)
        benchmarks.append(result)

    if args.gguf:
        result = benchmark_gguf(args.gguf, TEST_PROMPTS, args.max_tokens)
        benchmarks.append(result)

    if not any(benchmarks):
        print("No models specified. Use --original, --gptq, or --gguf")
        return

    generate_report(benchmarks, args.output)


if __name__ == "__main__":
    main()
