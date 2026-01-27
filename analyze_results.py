"""
Analyze and Visualize Benchmark Results
"""

import json
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load all benchmark results"""
    results = {}

    files = {
        "vllm": "vllm_results.json",
        "tgi": "tgi_results.json",
        "transformers": "transformers_results.json",
        "load_test": "load_test_results.json",
    }

    for key, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                results[key] = json.load(f)

    return results


def plot_latency_comparison(results: Dict[str, Any], output_path: str):
    """Plot latency comparison chart"""
    frameworks = []
    avg_latencies = []
    p50_latencies = []
    p99_latencies = []

    for framework in ["vllm", "tgi", "transformers"]:
        if framework in results and "statistics" in results[framework]:
            stats = results[framework]["statistics"]
            frameworks.append(framework.upper())
            avg_latencies.append(stats.get("avg_latency_ms", 0))
            p50_latencies.append(stats.get("p50_latency_ms", 0))
            p99_latencies.append(stats.get("p99_latency_ms", 0))

    if not frameworks:
        print("No latency data available")
        return

    x = np.arange(len(frameworks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, avg_latencies, width, label="Avg", color="#2196F3")
    bars2 = ax.bar(x, p50_latencies, width, label="P50", color="#4CAF50")
    bars3 = ax.bar(x + width, p99_latencies, width, label="P99", color="#FF9800")

    ax.set_xlabel("Framework")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_latency.png"), dpi=150)
    plt.close()


def plot_throughput_comparison(results: Dict[str, Any], output_path: str):
    """Plot throughput comparison"""
    frameworks = []
    tokens_per_sec = []

    for framework in ["vllm", "tgi", "transformers"]:
        if framework in results and "statistics" in results[framework]:
            stats = results[framework]["statistics"]
            frameworks.append(framework.upper())
            tokens_per_sec.append(stats.get("avg_tokens_per_sec", 0))

    if not frameworks:
        print("No throughput data available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(frameworks, tokens_per_sec, color=colors[:len(frameworks)])

    ax.set_xlabel("Framework")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Comparison")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_throughput.png"), dpi=150)
    plt.close()


def plot_load_test_results(results: Dict[str, Any], output_path: str):
    """Plot load test results"""
    if "load_test" not in results:
        print("No load test data available")
        return

    load_data = results["load_test"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for framework, color in [("vllm", "#2196F3"), ("tgi", "#4CAF50")]:
        if framework not in load_data:
            continue

        data = load_data[framework]
        concurrencies = [d["concurrency"] for d in data]
        rps = [d["requests_per_sec"] for d in data]
        p99_latencies = [d["p99_latency_ms"] for d in data]

        # RPS chart
        axes[0].plot(concurrencies, rps, marker="o", label=framework.upper(), color=color)

        # P99 Latency chart
        axes[1].plot(concurrencies, p99_latencies, marker="o", label=framework.upper(), color=color)

    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Requests/sec")
    axes[0].set_title("Throughput vs Concurrency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].set_title("P99 Latency vs Concurrency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_load_test.png"), dpi=150)
    plt.close()


def create_combined_chart(results: Dict[str, Any], output_path: str):
    """Create combined benchmark comparison chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Latency comparison
    frameworks = []
    latencies = {"avg": [], "p50": [], "p99": []}

    for fw in ["vllm", "tgi", "transformers"]:
        if fw in results and "statistics" in results[fw]:
            stats = results[fw]["statistics"]
            frameworks.append(fw.upper())
            latencies["avg"].append(stats.get("avg_latency_ms", 0))
            latencies["p50"].append(stats.get("p50_latency_ms", 0))
            latencies["p99"].append(stats.get("p99_latency_ms", 0))

    if frameworks:
        x = np.arange(len(frameworks))
        width = 0.25
        axes[0, 0].bar(x - width, latencies["avg"], width, label="Avg")
        axes[0, 0].bar(x, latencies["p50"], width, label="P50")
        axes[0, 0].bar(x + width, latencies["p99"], width, label="P99")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(frameworks)
        axes[0, 0].set_ylabel("Latency (ms)")
        axes[0, 0].set_title("Latency Comparison")
        axes[0, 0].legend()

    # 2. Throughput comparison
    throughputs = []
    fw_names = []
    for fw in ["vllm", "tgi", "transformers"]:
        if fw in results and "statistics" in results[fw]:
            fw_names.append(fw.upper())
            throughputs.append(results[fw]["statistics"].get("avg_tokens_per_sec", 0))

    if fw_names:
        axes[0, 1].bar(fw_names, throughputs, color=["#2196F3", "#4CAF50", "#FF9800"][:len(fw_names)])
        axes[0, 1].set_ylabel("Tokens/sec")
        axes[0, 1].set_title("Throughput Comparison")

    # 3. Load test - RPS
    if "load_test" in results:
        for fw, color in [("vllm", "#2196F3"), ("tgi", "#4CAF50")]:
            if fw in results["load_test"]:
                data = results["load_test"][fw]
                conc = [d["concurrency"] for d in data]
                rps = [d["requests_per_sec"] for d in data]
                axes[1, 0].plot(conc, rps, marker="o", label=fw.upper(), color=color)
        axes[1, 0].set_xlabel("Concurrency")
        axes[1, 0].set_ylabel("Requests/sec")
        axes[1, 0].set_title("Load Test: Throughput")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Load test - Latency
    if "load_test" in results:
        for fw, color in [("vllm", "#2196F3"), ("tgi", "#4CAF50")]:
            if fw in results["load_test"]:
                data = results["load_test"][fw]
                conc = [d["concurrency"] for d in data]
                lat = [d["p99_latency_ms"] for d in data]
                axes[1, 1].plot(conc, lat, marker="o", label=fw.upper(), color=color)
        axes[1, 1].set_xlabel("Concurrency")
        axes[1, 1].set_ylabel("P99 Latency (ms)")
        axes[1, 1].set_title("Load Test: P99 Latency")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Combined chart saved to {output_path}")


def generate_summary(results: Dict[str, Any], output_path: str):
    """Generate markdown summary"""
    lines = ["# LLM Serving Benchmark Results\n"]

    # Framework comparison table
    lines.append("## Framework Comparison\n")
    lines.append("| Framework | Avg Latency (ms) | P99 Latency (ms) | Tokens/sec |")
    lines.append("|-----------|------------------|------------------|------------|")

    for fw in ["vllm", "tgi", "transformers"]:
        if fw in results and "statistics" in results[fw]:
            stats = results[fw]["statistics"]
            lines.append(
                f"| {fw.upper()} | {stats.get('avg_latency_ms', 0):.2f} | "
                f"{stats.get('p99_latency_ms', 0):.2f} | "
                f"{stats.get('avg_tokens_per_sec', 0):.1f} |"
            )

    # Load test results
    if "load_test" in results:
        lines.append("\n## Load Test Results\n")
        lines.append("| Framework | Concurrency | RPS | P99 Latency (ms) |")
        lines.append("|-----------|-------------|-----|------------------|")

        for fw in ["vllm", "tgi"]:
            if fw in results["load_test"]:
                for data in results["load_test"][fw]:
                    lines.append(
                        f"| {fw.upper()} | {data['concurrency']} | "
                        f"{data['requests_per_sec']:.2f} | "
                        f"{data['p99_latency_ms']:.2f} |"
                    )

    # Conclusions
    lines.append("\n## Observations\n")
    lines.append("- vLLM generally provides better throughput for high-concurrency scenarios")
    lines.append("- TGI offers competitive performance with simpler deployment")
    lines.append("- Direct transformers inference is slower but requires no additional infrastructure")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Summary saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/benchmark_comparison.png")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if not results:
        print("No results found. Run benchmarks first.")
        return

    create_combined_chart(results, args.output)
    generate_summary(results, os.path.join(args.results_dir, "summary.md"))


if __name__ == "__main__":
    main()
