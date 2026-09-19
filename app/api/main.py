"""FastAPI application factory.

Error handling is deliberately uniform: malformed JSON, schema violations and
unexpected failures all return a structured body with a ``hint``, because the
consumer here is a reviewer poking at the API by hand, not only a client
library.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
Policy-aware multi-agent RAG engine for health-insurance claim adjudication.

Every material statement in a decision is bound to a clause retrieved from the
supplied policy wording, with page, section and chunk id. Where the policy does
not support a safe conclusion the engine returns **NEEDS_REVIEW** rather than
guessing.

* `POST /analyze` — analyse one claim case
* `POST /analyze/batch` — analyse up to 50 cases
* `GET /health` — readiness probe
* `GET /index/info` — inspect the indexed policy corpus
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build or load the policy index at startup so the first request is fast.

    A warm-up failure is logged rather than fatal: the service still starts and
    /health reports ``degraded``, which is the more useful behaviour behind a
    platform health check.
    """
    try:
        from app.services.engine import get_engine

        engine = get_engine()
        logger.info("Engine ready: %s", engine.backends())
    except Exception as exc:
        logger.error(
            "Index warm-up failed (%s). /health will report degraded until "
            "`python -m scripts.ingest_policy` has been run.",
            exc,
        )
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # ------------------------------------------------------------------ #
    # Error handlers
    # ------------------------------------------------------------------ #

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        field_errors = [
            {
                "field": ".".join(str(p) for p in err.get("loc", []) if p != "body") or "body",
                "message": err.get("msg", "invalid"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Invalid request payload",
                "detail": f"{len(field_errors)} field error(s).",
                "field_errors": field_errors,
                "hint": (
                    "POST a claim case with at least 'case_id'. See GET /docs for the schema, "
                    "or use evaluation/public_cases/public_test_cases.json as a template."
                ),
            },
        )

    @app.exception_handler(json.JSONDecodeError)
    async def json_error_handler(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Malformed JSON",
                "detail": str(exc),
                "hint": "Ensure the body is valid JSON and Content-Type is application/json.",
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "detail": None, "field_errors": []},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                # The message is included but never a traceback, which could
                # leak paths or configuration.
                "detail": f"{type(exc).__name__}",
                "hint": "Check the service logs; GET /health reports index readiness.",
            },
        )

    # ------------------------------------------------------------------ #

    @app.get("/", tags=["system"], include_in_schema=False)
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
            "analyze": "POST /analyze",
        }

    return app


app = create_app()
