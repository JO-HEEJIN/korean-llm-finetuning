"""
Health check utilities for LLM server
"""

import asyncio
import httpx
from typing import Dict, Any


async def check_vllm_health(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Check vLLM backend health"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code
            }
    except httpx.RequestError as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_model_loaded(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Check if model is loaded and ready"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                return {
                    "status": "ready",
                    "models": models.get("data", [])
                }
            return {
                "status": "not_ready",
                "status_code": response.status_code
            }
    except httpx.RequestError as e:
        return {
            "status": "not_ready",
            "error": str(e)
        }


async def full_health_check(vllm_url: str) -> Dict[str, Any]:
    """Perform full health check"""
    vllm_health = await check_vllm_health(vllm_url)
    model_status = await check_model_loaded(vllm_url)

    overall_healthy = (
        vllm_health.get("status") == "healthy" and
        model_status.get("status") == "ready"
    )

    return {
        "overall": "healthy" if overall_healthy else "unhealthy",
        "vllm_backend": vllm_health,
        "model": model_status
    }


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    result = asyncio.run(full_health_check(url))

    print(f"Health Check Results for {url}:")
    print(f"  Overall: {result['overall']}")
    print(f"  vLLM Backend: {result['vllm_backend']['status']}")
    print(f"  Model: {result['model']['status']}")

    sys.exit(0 if result['overall'] == 'healthy' else 1)
