"""FastAPI application entrypoint for chatmem local service."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yaml

from chatmem.api.auth import TokenVerifier, get_or_create_auth_token
from chatmem.api.models import (
    ApiConfig,
    ContextRequest,
    ContextResponse,
    DeleteResponse,
    HealthResponse,
    IngestMessagesRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from chatmem.api.service import ChatmemService
from chatmem.retrieval.pipeline import QdrantConfig
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(
    config: ApiConfig | None = None,
    config_path: str | Path | None = None,
    qdrant_config: QdrantConfig | None = None,
    db_path: str | Path | None = None,
    service: ChatmemService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    # 1. Resolve configuration
    api_cfg = config
    qdrant_cfg = qdrant_config

    if api_cfg is None:
        if config_path is not None:
            cfg_p = Path(config_path).expanduser().resolve()
            if not cfg_p.exists():
                raise FileNotFoundError(f"Configuration file not found: {cfg_p}")
            with open(cfg_p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            api_cfg = ApiConfig(**raw.get("api", {}))
            if qdrant_cfg is None and "qdrant" in raw:
                qdrant_cfg = QdrantConfig(**raw["qdrant"])
        else:
            default_p = Path("configs/api.yaml")
            if default_p.exists():
                with open(default_p, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                api_cfg = ApiConfig(**raw.get("api", {}))
                if qdrant_cfg is None and "qdrant" in raw:
                    qdrant_cfg = QdrantConfig(**raw["qdrant"])
            else:
                api_cfg = ApiConfig()

    # 2. Host interface security validation
    if (api_cfg.host == "0.0.0.0" or api_cfg.host == "::") and not api_cfg.allow_all_interfaces:
        raise ValueError(
            "Refusing to bind to all interfaces (0.0.0.0). "
            "Chatmem stores private conversational history on localhost only. "
            "Set allow_all_interfaces: true in config if this is explicitly intended."
        )

    # 3. Setup auth token & verifier
    token = get_or_create_auth_token(api_cfg.auth_token_path)
    verifier = TokenVerifier(token)

    # 4. Lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_service = False
        if not hasattr(app.state, "service") or app.state.service is None:
            owns_service = True
            logger.info("Initializing chatmem service...")
            try:
                srv = ChatmemService(
                    api_config=api_cfg,
                    qdrant_config=qdrant_cfg,
                    db_path=db_path,
                )
                srv.warmup()
                app.state.service = srv
            except Exception as e:
                err_msg = str(e)
                if "locked" in err_msg.lower() or "already accessed" in err_msg.lower() or "resource temporarily unavailable" in err_msg.lower():
                    logger.error(
                        "Storage is locked by another process: %s. "
                        "Embedded Qdrant allows only one active process.",
                        e,
                    )
                raise
        yield
        if owns_service and hasattr(app.state, "service") and app.state.service:
            logger.info("Shutting down chatmem service...")
            app.state.service.close()

    app = FastAPI(
        title="chatmem local API",
        version="0.1.0",
        description="Localhost REST API for privacy-preserving personal conversation retrieval.",
        lifespan=lifespan,
    )

    if service is not None:
        app.state.service = service

    # 5. Middlewares
    # CORS
    if "*" in api_cfg.allowed_cors_origins:
        raise ValueError("Wildcard '*' CORS origin is not permitted in chatmem for security reasons.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_cfg.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Request size limiter
    @app.middleware("http")
    async def request_size_limiter(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > api_cfg.max_request_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "detail": f"Request payload exceeds maximum allowed size ({api_cfg.max_request_bytes} bytes)."
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)

    # Privacy-preserving access logger
    @app.middleware("http")
    async def privacy_access_logger(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result_count = response.headers.get("X-Result-Count", "-")
        # Strict privacy rule: Log ONLY method, path, status, latency ms, result count.
        # NEVER log request body, query params, or result texts.
        logger.info(
            "method=%s path=%s status=%d latency_ms=%.2f results=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            result_count,
        )
        return response

    # 6. Endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        srv: ChatmemService = request.app.state.service
        return srv.get_health()

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(verifier)])
    async def search(req: SearchRequest, request: Request, response: Response) -> SearchResponse:
        srv: ChatmemService = request.app.state.service
        res = await srv.search(req)
        response.headers["X-Result-Count"] = str(len(res.results))
        return res

    @app.post("/context", response_model=ContextResponse, dependencies=[Depends(verifier)])
    async def context(req: ContextRequest, request: Request, response: Response) -> ContextResponse:
        srv: ChatmemService = request.app.state.service
        res = await srv.format_context(
            query=req.query,
            mode=req.mode,
            max_chars=req.max_chars,
        )
        response.headers["X-Result-Count"] = str(res.chunks_used)
        return res

    @app.post("/ingest/messages", response_model=IngestResponse, dependencies=[Depends(verifier)])
    async def ingest_messages(req: IngestMessagesRequest, request: Request) -> IngestResponse:
        srv: ChatmemService = request.app.state.service
        try:
            chunks_indexed = await srv.ingest_messages(req)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        return IngestResponse(
            status="ok",
            conversation_id=req.conversation_id,
            platform=req.platform,
            chunks_indexed=chunks_indexed,
            messages_indexed=len(req.messages),
        )

    @app.delete(
        "/conversations/{platform}/{conversation_id}",
        response_model=DeleteResponse,
        dependencies=[Depends(verifier)],
    )
    async def delete_conversation(platform: str, conversation_id: str, request: Request) -> DeleteResponse:
        srv: ChatmemService = request.app.state.service
        chunks_deleted = await srv.delete_conversation(conversation_id=conversation_id, platform=platform)
        return DeleteResponse(
            status="deleted",
            conversation_id=conversation_id,
            platform=platform,
            chunks_deleted=chunks_deleted,
        )

    return app
