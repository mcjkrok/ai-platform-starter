import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from .config import settings
from .llm_client import LLMClient, LLMError, MockLLMClient

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A small internal AI platform: gateway to an LLM backend "
    "with health checks and Prometheus metrics.",
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total requests forwarded to the LLM backend",
    ["outcome"],
)
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "Latency of LLM backend calls in seconds",
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    # route.path (e.g. /api/chat) instead of raw URL keeps label cardinality low
    route = request.scope.get("route")
    path = route.path if route else request.url.path
    HTTP_REQUESTS.labels(request.method, path, response.status_code).inc()
    return response


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    latency_ms: int


def get_llm_client() -> LLMClient:
    if settings.llm_mock:
        return MockLLMClient()
    return LLMClient()


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    """Liveness probe: the process is up and can serve requests."""
    return {"status": "ok", "version": settings.app_version}


@app.get("/readyz", tags=["ops"])
def readyz() -> dict:
    """Readiness probe: the app is ready to accept traffic."""
    return {"status": "ready", "mock_mode": settings.llm_mock}


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/chat", response_model=ChatResponse, tags=["ai"])
async def chat(
    request: ChatRequest,
    client: Annotated[LLMClient, Depends(get_llm_client)],
) -> ChatResponse:
    started = time.perf_counter()
    try:
        result = await client.chat(request.prompt, request.model)
    except LLMError as exc:
        LLM_REQUESTS.labels(outcome="error").inc()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started
    LLM_LATENCY.observe(elapsed)
    LLM_REQUESTS.labels(outcome="success").inc()

    return ChatResponse(
        reply=result["reply"],
        model=result["model"],
        latency_ms=int(elapsed * 1000),
    )
