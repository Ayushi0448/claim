"""Engine facade.

One entry point used by the API, the evaluation harness and the tests, so all
three exercise exactly the same code path. Failures inside the workflow are
converted into a valid NEEDS_REVIEW response rather than an exception: a claims
system that crashes tells the reviewer nothing, whereas an abstention with the
error recorded in the trace is actionable.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings, get_settings
from app.graph.workflow import ClaimWorkflow
from app.models.schemas import (
    AnalysisResponse,
    ClaimCase,
    Decision,
    TraceEvent,
    ValidationReport,
    ValidationStatus,
)
from app.retrieval.index import HybridRetriever, get_retriever
from app.services.llm import get_llm_client

logger = logging.getLogger(__name__)


class ClaimDecisionEngine:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever(self.settings)
        self.workflow = ClaimWorkflow(self.retriever, self.settings)

    # ------------------------------------------------------------------ #

    def backends(self) -> dict[str, str]:
        info = self.retriever.backend_info()
        info["orchestrator"] = self.workflow.orchestrator
        info["llm"] = get_llm_client().description
        info["engine_version"] = self.settings.version
        return info

    # ------------------------------------------------------------------ #

    def analyze(self, case: ClaimCase) -> AnalysisResponse:
        started = time.perf_counter()
        try:
            state = self.workflow.run(case)
            response: AnalysisResponse | None = state.get("decision_draft")
            if response is None:
                raise RuntimeError("Workflow produced no decision")

            response.trace = list(state.get("trace", []))
            response.retrieval_queries = list(state.get("queries", []))
            response.backends = self.backends()
            response.total_elapsed_ms = int((time.perf_counter() - started) * 1000)

            errors = state.get("errors") or []
            if errors:
                response.trace.append(
                    TraceEvent(
                        agent="engine",
                        action="Recoverable errors during analysis",
                        status="WARNING",
                        metrics={"errors": errors[:5]},
                    )
                )
            return response

        except Exception as exc:
            logger.exception("Analysis failed for case %s", case.case_id)
            elapsed = int((time.perf_counter() - started) * 1000)
            return AnalysisResponse(
                case_id=case.case_id,
                decision=Decision.NEEDS_REVIEW,
                confidence=0.0,
                summary=(
                    f"Claim {case.case_id} could not be analysed because the decision pipeline "
                    "failed. The claim is referred for manual review."
                ),
                abstained=True,
                abstain_reason=(
                    f"INSUFFICIENT_EVIDENCE: analysis pipeline error ({type(exc).__name__})."
                ),
                validation=ValidationReport(
                    status=ValidationStatus.FAIL,
                    unsupported_claims=["Pipeline error prevented verification."],
                    revision_required=False,
                    checks_run=[],
                    checks_failed=["pipeline"],
                    notes=[f"{type(exc).__name__}: {exc}"[:300]],
                ),
                backends=self._safe_backends(),
                trace=[
                    TraceEvent(
                        agent="engine",
                        action="Analysis aborted",
                        status="ERROR",
                        elapsed_ms=elapsed,
                        metrics={"error_type": type(exc).__name__},
                    )
                ],
                engine_version=self.settings.version,
                total_elapsed_ms=elapsed,
            )

    def _safe_backends(self) -> dict[str, str]:
        try:
            return self.backends()
        except Exception:  # pragma: no cover - defensive
            return {"engine_version": self.settings.version}


_ENGINE: ClaimDecisionEngine | None = None


def get_engine(settings: Settings | None = None) -> ClaimDecisionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ClaimDecisionEngine(settings=settings)
    return _ENGINE


def reset_engine() -> None:
    """Test hook."""
    global _ENGINE
    _ENGINE = None
