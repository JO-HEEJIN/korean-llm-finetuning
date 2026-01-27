"""
Configuration for On-Premise LLM Server
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model settings
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    model_path: Optional[str] = os.getenv("MODEL_PATH", None)

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # vLLM settings
    tensor_parallel_size: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "4096"))

    # API settings
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))

    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

    # Authentication
    api_key: Optional[str] = os.getenv("API_KEY", None)
    enable_auth: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_requests: bool = os.getenv("LOG_REQUESTS", "true").lower() == "true"

    # vLLM backend URL (when using as proxy)
    vllm_backend_url: str = os.getenv("VLLM_BACKEND_URL", "http://localhost:8001")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# Model configurations for different use cases
MODEL_CONFIGS = {
    "small": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
    },
    "medium": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
    },
    "large": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.95,
        "tensor_parallel_size": 2,
    },
}
