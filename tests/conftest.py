"""Shared fixtures.

The retriever is session-scoped: building the index costs a few seconds and is
pure, so every test shares one instance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.models.schemas import ClaimCase  # noqa: E402
from app.retrieval.chunking import chunk_policy  # noqa: E402
from app.retrieval.index import HybridRetriever  # noqa: E402
from app.retrieval.ingestion import load_policy_pages  # noqa: E402
from app.services.engine import ClaimDecisionEngine  # noqa: E402

PUBLIC_CASES_PATH = PROJECT_ROOT / "evaluation" / "public_cases" / "public_test_cases.json"
CUSTOM_CASES_PATH = PROJECT_ROOT / "evaluation" / "custom_cases" / "custom_test_cases.json"
EXPECTED_PUBLIC_PATH = PROJECT_ROOT / "evaluation" / "expected_results" / "expected_public.json"
EXPECTED_CUSTOM_PATH = PROJECT_ROOT / "evaluation" / "expected_results" / "expected_custom.json"


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def policy_pages(settings):
    return load_policy_pages(settings.policy_pdf)


@pytest.fixture(scope="session")
def chunks(policy_pages, settings):
    return chunk_policy(policy_pages, source_name=settings.policy_source_name)


@pytest.fixture(scope="session")
def retriever() -> HybridRetriever:
    return HybridRetriever().ensure_ready()


@pytest.fixture(scope="session")
def engine(retriever) -> ClaimDecisionEngine:
    return ClaimDecisionEngine(retriever=retriever)


@pytest.fixture(scope="session")
def public_cases() -> list[dict]:
    return json.loads(PUBLIC_CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def custom_cases() -> list[dict]:
    return json.loads(CUSTOM_CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def expected_public() -> dict:
    return json.loads(EXPECTED_PUBLIC_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="session")
def expected_custom() -> dict:
    return json.loads(EXPECTED_CUSTOM_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="session")
def case_by_id(public_cases, custom_cases) -> dict[str, dict]:
    return {c["case_id"]: c for c in [*public_cases, *custom_cases]}


@pytest.fixture(scope="session")
def analyses(engine, public_cases, custom_cases) -> dict:
    """Every case analysed once; tests assert against the cached results."""
    results = {}
    for raw in [*public_cases, *custom_cases]:
        results[raw["case_id"]] = engine.analyze(ClaimCase(**raw))
    return results


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    with TestClient(app) as client:
        yield client
