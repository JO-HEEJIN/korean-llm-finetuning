"""
Concurrent Load Test for LLM Serving
"""

import time
import asyncio
import json
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass, asdict, field

import aiohttp


@dataclass
class RequestResult:
    success: bool
    latency_ms: float
    status_code: int = 0
    error: str = ""


@dataclass
class LoadTestResult:
    framework: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    requests_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    errors: List[str] = field(default_factory=list)


class LoadTester:
    def __init__(self, base_url: str, framework: str = "unknown"):
        self.base_url = base_url
        self.framework = framework

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int = 128
    ) -> RequestResult:
        """Send a single request"""
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        start_time = time.perf_counter()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                await response.text()
                latency_ms = (time.perf_counter() - start_time) * 1000
                return RequestResult(
                    success=response.status == 200,
                    latency_ms=latency_ms,
                    status_code=response.status
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )

    async def run_load_test(
        self,
        concurrency: int,
        total_requests: int,
        max_tokens: int = 128
    ) -> LoadTestResult:
        """Run concurrent load test"""
        prompts = [
            "Explain machine learning briefly.",
            "What is artificial intelligence?",
            "Describe cloud computing.",
            "How do neural networks work?",
            "What is deep learning?",
        ]

        semaphore = asyncio.Semaphore(concurrency)
        results: List[RequestResult] = []

        async def bounded_request(session: aiohttp.ClientSession, prompt: str):
            async with semaphore:
                return await self.send_request(session, prompt, max_tokens)

        print(f"Running load test: {concurrency} concurrent, {total_requests} total requests")

        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded_request(session, prompts[i % len(prompts)])
                for i in range(total_requests)
            ]
            results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            latencies = [r.latency_ms for r in successful]
            latencies.sort()

            return LoadTestResult(
                framework=self.framework,
                concurrency=concurrency,
                total_requests=total_requests,
                successful_requests=len(successful),
                failed_requests=len(failed),
                total_time_sec=total_time,
                requests_per_sec=len(successful) / total_time,
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=latencies[len(latencies) // 2],
                p90_latency_ms=latencies[int(len(latencies) * 0.9)],
                p99_latency_ms=latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                errors=[r.error for r in failed if r.error][:10]
            )
        else:
            return LoadTestResult(
                framework=self.framework,
                concurrency=concurrency,
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=len(failed),
                total_time_sec=total_time,
                requests_per_sec=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p90_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                errors=[r.error for r in failed if r.error][:10]
            )


async def run_benchmark_suite(
    vllm_url: str = "http://localhost:8001",
    tgi_url: str = "http://localhost:8002",
    concurrency_levels: List[int] = None,
    requests_per_level: int = 100
) -> Dict[str, List[LoadTestResult]]:
    """Run load tests on multiple frameworks"""
    if concurrency_levels is None:
        concurrency_levels = [10, 50, 100]

    results = {"vllm": [], "tgi": []}

    frameworks = [
        ("vllm", vllm_url),
        ("tgi", tgi_url),
    ]

    for framework, url in frameworks:
        print(f"\n{'='*50}")
        print(f"Testing {framework.upper()}")
        print(f"{'='*50}")

        tester = LoadTester(url, framework)

        for concurrency in concurrency_levels:
            print(f"\nConcurrency: {concurrency}")
            result = await tester.run_load_test(concurrency, requests_per_level)
            results[framework].append(result)

            print(f"  Success: {result.successful_requests}/{result.total_requests}")
            print(f"  RPS: {result.requests_per_sec:.2f}")
            print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
            print(f"  P99 Latency: {result.p99_latency_ms:.2f}ms")

    return results


def save_results(results: Dict[str, List[LoadTestResult]], filepath: str):
    """Save load test results"""
    data = {
        framework: [asdict(r) for r in framework_results]
        for framework, framework_results in results.items()
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--tgi-url", default="http://localhost:8002")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", default="results/load_test_results.json")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark_suite(
        vllm_url=args.vllm_url,
        tgi_url=args.tgi_url,
        concurrency_levels=args.concurrency,
        requests_per_level=args.requests
    ))

    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
