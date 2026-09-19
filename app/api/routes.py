"""HTTP routes."""

from __future__ import annotations

import logging
import time
from collections import Counter

from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    IndexInfoResponse,
    PolicyChunkSummary,
)
from app.config import get_settings
from app.models.schemas import AnalysisResponse, ClaimCase
from app.retrieval.index import get_retriever
from app.services.engine import get_engine

logger = logging.getLogger(__name__)
router = APIRouter()

_STARTED_AT = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health and readiness probe",
)
def health() -> HealthResponse:
    """Readiness probe.

    Returns 200 with ``status="degraded"`` rather than an error when the index
    is missing, so a platform health check can distinguish "process is up but
    not yet ready" from "process is broken".
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        ready = retriever.ready
        n_chunks = len(retriever.chunks)
        backends = retriever.backend_info()
        detail = None
    except Exception as exc:
        logger.warning("Health check could not initialise the index: %s", exc)
        ready, n_chunks, backends = False, 0, {}
        detail = f"Policy index unavailable: {type(exc).__name__}"

    return HealthResponse(
        status="ok" if ready else "degraded",
        version=settings.version,
        index_ready=ready,
        policy_indexed=ready and n_chunks > 0,
        chunks_indexed=n_chunks,
        policy_source=settings.policy_source_name,
        policy_id=settings.policy_id,
        backends=backends,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
        detail=detail,
    )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    response_model_exclude_none=False,
    tags=["analysis"],
    summary="Analyze one claim case and return a structured decision",
)
async def analyze(request: Request, payload: AnalyzeRequest) -> AnalysisResponse:
    settings = get_settings()

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_request_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body exceeds {settings.max_request_bytes} bytes.",
        )

    try:
        case: ClaimCase = payload.resolve_case()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid claim case: {exc}. A claim requires at least 'case_id'; "
                "see /docs for the full schema."
            ),
        ) from exc

    try:
        engine = get_engine()
    except Exception as exc:
        logger.exception("Engine unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Policy index is not available. Run `python -m scripts.ingest_policy` "
                f"and restart the service. ({type(exc).__name__})"
            ),
        ) from exc

    # The engine converts internal failures into a NEEDS_REVIEW response rather
    # than raising, so a reviewer always receives a valid, auditable decision.
    result = engine.analyze(case)

    if not payload.include_evidence:
        result.evidence = []
    if not payload.include_trace:
        result.trace = []
    return result


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    tags=["analysis"],
    summary="Analyze several claim cases in one call",
)
def analyze_batch(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    engine = get_engine()
    results: list[AnalysisResponse] = []
    for case in payload.cases:
        result = engine.analyze(case)
        if not payload.include_evidence:
            result.evidence = []
        if not payload.include_trace:
            result.trace = []
        results.append(result)
    return BatchAnalyzeResponse(results=results, count=len(results))


@router.get(
    "/index/info",
    response_model=IndexInfoResponse,
    tags=["system"],
    summary="Inspect the indexed policy corpus",
)
def index_info() -> IndexInfoResponse:
    """Lets a reviewer verify chunking quality without reading the code."""
    try:
        retriever = get_retriever()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Policy index unavailable: {type(exc).__name__}",
        ) from exc

    sections = Counter(c.section for c in retriever.chunks)
    return IndexInfoResponse(
        n_chunks=len(retriever.chunks),
        sections=dict(sections),
        pages=sorted({c.page for c in retriever.chunks}),
        backends=retriever.backend_info(),
        sample=[
            PolicyChunkSummary(
                chunk_id=c.chunk_id,
                page=c.page,
                section=c.section,
                heading=c.heading,
                char_count=c.char_count,
            )
            for c in retriever.chunks[:25]
        ],
    )
