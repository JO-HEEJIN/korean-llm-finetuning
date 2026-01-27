"""
Transformers Direct Inference Benchmark
"""

import time
import json
import gc
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    framework: str = "transformers"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    tokens_per_sec: float = 0
    gpu_memory_mb: float = 0


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


class TransformersBenchmark:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results: List[BenchmarkResult] = []
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        print(f"Model loaded. GPU Memory: {get_gpu_memory_mb():.2f} MB")

    def single_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> BenchmarkResult:
        """Single request benchmark"""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        gpu_memory_before = get_gpu_memory_mb()

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        output_length = outputs.shape[1] - input_length
        tokens_per_sec = output_length / (latency_ms / 1000) if latency_ms > 0 else 0

        result = BenchmarkResult(
            framework="transformers",
            input_tokens=input_length,
            output_tokens=output_length,
            latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
            gpu_memory_mb=get_gpu_memory_mb() - gpu_memory_before,
        )
        self.results.append(result)
        return result

    def throughput_test(
        self,
        prompts: List[str],
        max_tokens: int = 128
    ) -> Dict[str, float]:
        """Throughput benchmark (sequential for transformers)"""
        total_tokens = 0
        total_requests = 0

        start_time = time.perf_counter()

        for prompt in prompts:
            try:
                result = self.single_request(prompt, max_tokens)
                total_tokens += result.output_tokens
                total_requests += 1
            except Exception as e:
                print(f"Error: {e}")
                continue

        end_time = time.perf_counter()
        total_time = end_time - start_time

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
            "requests_per_sec": total_requests / total_time if total_time > 0 else 0,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics"""
        if not self.results:
            return {}

        latencies = [r.latency_ms for r in self.results]
        latencies.sort()

        return {
            "framework": "transformers",
            "total_requests": len(self.results),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p90_latency_ms": latencies[int(len(latencies) * 0.9)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_tokens_per_sec": sum(r.tokens_per_sec for r in self.results) / len(self.results),
            "avg_gpu_memory_mb": sum(r.gpu_memory_mb for r in self.results) / len(self.results),
        }

    def save_results(self, filepath: str):
        """Save results to JSON"""
        data = {
            "results": [asdict(r) for r in self.results],
            "statistics": self.get_statistics()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_requests: int = 50,
    max_tokens: int = 128
):
    """Run full transformers benchmark"""
    benchmark = TransformersBenchmark(model_name)

    prompts = [
        "Explain what machine learning is in simple terms.",
        "Write a short poem about technology.",
        "What are the benefits of cloud computing?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]

    print(f"Running Transformers benchmark with {num_requests} requests...")

    print("\n1. Single Request Latency Test")
    for i in range(min(10, num_requests)):
        result = benchmark.single_request(prompts[i], max_tokens)
        print(f"  Request {i+1}: {result.latency_ms:.2f}ms, {result.tokens_per_sec:.1f} tokens/s")

    print("\n2. Throughput Test")
    throughput = benchmark.throughput_test(prompts[10:], max_tokens)
    print(f"  Requests/sec: {throughput['requests_per_sec']:.2f}")
    print(f"  Tokens/sec: {throughput['tokens_per_sec']:.2f}")

    print("\n3. Statistics")
    stats = benchmark.get_statistics()
    print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P50 Latency: {stats['p50_latency_ms']:.2f}ms")
    print(f"  P99 Latency: {stats['p99_latency_ms']:.2f}ms")

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="results/transformers_results.json")
    args = parser.parse_args()

    benchmark = run_benchmark(args.model, args.requests, args.max_tokens)
    benchmark.save_results(args.output)
    benchmark.cleanup()
    print(f"\nResults saved to {args.output}")
