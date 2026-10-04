"""
Text Generation Inference (TGI) Serving Benchmark
"""

import time
import asyncio
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import httpx


@dataclass
class BenchmarkResult:
    framework: str = "tgi"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    tokens_per_sec: float = 0
    time_to_first_token_ms: float = 0


class TGIBenchmark:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []

    async def single_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> BenchmarkResult:
        """Single request latency benchmark"""
        async with httpx.AsyncClient(timeout=300) as client:
            # TGI generate endpoint
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.01,
                    "do_sample": temperature > 0,
                }
            }

            start_time = time.perf_counter()
            response = await client.post(
                f"{self.base_url}/generate",
                json=payload
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("generated_text", "")
                # Estimate tokens (rough approximation)
                output_tokens = len(generated_text.split()) * 1.3
                input_tokens = len(prompt.split()) * 1.3

                tokens_per_sec = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

                result = BenchmarkResult(
                    framework="tgi",
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                    tokens_per_sec=tokens_per_sec,
                )
                self.results.append(result)
                return result
            else:
                raise Exception(f"Request failed: {response.status_code}")

    async def single_request_openai(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> BenchmarkResult:
        """Single request using OpenAI-compatible endpoint"""
        async with httpx.AsyncClient(timeout=300) as client:
            payload = {
                "model": "tgi",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            start_time = time.perf_counter()
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                output_tokens = usage.get("completion_tokens", 0)
                input_tokens = usage.get("prompt_tokens", 0)

                tokens_per_sec = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

                result = BenchmarkResult(
                    framework="tgi",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    tokens_per_sec=tokens_per_sec,
                )
                self.results.append(result)
                return result
            else:
                raise Exception(f"Request failed: {response.status_code}")

    async def throughput_test(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        batch_size: int = 10
    ) -> Dict[str, float]:
        """Throughput benchmark"""
        total_tokens = 0
        total_requests = 0

        start_time = time.perf_counter()

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tasks = [
                self.single_request(prompt, max_tokens)
                for prompt in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, BenchmarkResult):
                    total_tokens += result.output_tokens
                    total_requests += 1

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
            "framework": "tgi",
            "total_requests": len(self.results),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p90_latency_ms": latencies[int(len(latencies) * 0.9)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_tokens_per_sec": sum(r.tokens_per_sec for r in self.results) / len(self.results),
        }

    def save_results(self, filepath: str):
        """Save results to JSON"""
        data = {
            "results": [asdict(r) for r in self.results],
            "statistics": self.get_statistics()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


async def run_benchmark(
    base_url: str = "http://localhost:8002",
    num_requests: int = 100,
    max_tokens: int = 128
):
    """Run full TGI benchmark"""
    benchmark = TGIBenchmark(base_url)

    prompts = [
        "Explain what machine learning is in simple terms.",
        "Write a short poem about technology.",
        "What are the benefits of cloud computing?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]

    print(f"Running TGI benchmark with {num_requests} requests...")

    print("\n1. Single Request Latency Test")
    for i in range(min(10, num_requests)):
        result = await benchmark.single_request(prompts[i], max_tokens)
        print(f"  Request {i+1}: {result.latency_ms:.2f}ms, {result.tokens_per_sec:.1f} tokens/s")

    print("\n2. Throughput Test")
    throughput = await benchmark.throughput_test(prompts, max_tokens, batch_size=10)
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
    parser.add_argument("--url", default="http://localhost:8002")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="results/tgi_results.json")
    args = parser.parse_args()

    benchmark = asyncio.run(run_benchmark(args.url, args.requests, args.max_tokens))
    benchmark.save_results(args.output)
    print(f"\nResults saved to {args.output}")
