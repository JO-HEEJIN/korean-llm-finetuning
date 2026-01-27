"""
FastAPI wrapper for vLLM with authentication, logging, and rate limiting
OpenAI-compatible API endpoints
"""

import time
import uuid
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import settings

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
TOKEN_COUNT = Counter(
    "llm_tokens_total",
    "Total number of tokens",
    ["type"]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Request/Response Models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = Field(default="default")
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# HTTP client for vLLM backend
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=settings.vllm_backend_url,
        timeout=httpx.Timeout(settings.request_timeout)
    )
    logger.info(f"Connected to vLLM backend: {settings.vllm_backend_url}")
    yield
    await http_client.aclose()


app = FastAPI(
    title="On-Premise LLM Server",
    description="OpenAI-compatible LLM serving API",
    version="1.0.0",
    lifespan=lifespan
)

# Add middlewares
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication dependency
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not settings.enable_auth:
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="API key required")

    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if settings.log_requests:
        logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    if settings.log_requests:
        logger.info(f"[{request_id}] Completed in {duration:.3f}s - Status: {response.status_code}")

    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        response = await http_client.get("/health")
        backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False

    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend": "connected" if backend_healthy else "disconnected",
        "timestamp": int(time.time())
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}seconds")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: bool = Depends(verify_api_key)
):
    """OpenAI-compatible chat completions endpoint"""
    start_time = time.time()

    try:
        # Forward to vLLM backend
        response = await http_client.post(
            "/v1/chat/completions",
            json=body.model_dump()
        )

        if response.status_code != 200:
            REQUEST_COUNT.labels(endpoint="chat_completions", status="error").inc()
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()

        # Update metrics
        REQUEST_COUNT.labels(endpoint="chat_completions", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="chat_completions").observe(time.time() - start_time)

        if "usage" in result:
            TOKEN_COUNT.labels(type="prompt").inc(result["usage"].get("prompt_tokens", 0))
            TOKEN_COUNT.labels(type="completion").inc(result["usage"].get("completion_tokens", 0))

        return result

    except httpx.RequestError as e:
        REQUEST_COUNT.labels(endpoint="chat_completions", status="error").inc()
        logger.error(f"Backend request failed: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")


@app.post("/v1/completions", response_model=CompletionResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}seconds")
async def completions(
    request: Request,
    body: CompletionRequest,
    _: bool = Depends(verify_api_key)
):
    """OpenAI-compatible completions endpoint"""
    start_time = time.time()

    try:
        response = await http_client.post(
            "/v1/completions",
            json=body.model_dump()
        )

        if response.status_code != 200:
            REQUEST_COUNT.labels(endpoint="completions", status="error").inc()
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()

        REQUEST_COUNT.labels(endpoint="completions", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="completions").observe(time.time() - start_time)

        if "usage" in result:
            TOKEN_COUNT.labels(type="prompt").inc(result["usage"].get("prompt_tokens", 0))
            TOKEN_COUNT.labels(type="completion").inc(result["usage"].get("completion_tokens", 0))

        return result

    except httpx.RequestError as e:
        REQUEST_COUNT.labels(endpoint="completions", status="error").inc()
        logger.error(f"Backend request failed: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")


@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_api_key)):
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
