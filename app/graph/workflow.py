"""LangGraph orchestration.

    case_analysis → policy_evidence → coverage_exclusion → decision → validation
                                                              ↑           │
                                                              └── revise ──┘
                                                                    (max N)

Each node is a thin adapter: it reads typed objects from the state, calls the
agent that owns that responsibility, and writes typed objects back. Nodes never
reach into another agent's internals, which is what makes the boundaries real
rather than decorative.

The conditional edge after validation is the retry behaviour required by the
rubric: a FAIL routes back to the Decision Agent, which re-scores with the
validation result folded into confidence. A second FAIL terminates in
NEEDS_REVIEW rather than shipping an unverified decision.

If ``langgraph`` is unavailable the same graph is executed by a small built-in
runner with identical semantics, so the system stays deployable on a minimal
free-tier image.
"""

from __future__ import annotations

import logging

from app.agents.base import AgentTimer
from app.agents.case_analysis import CaseAnalysisAgent
from app.agents.coverage_exclusion import CoverageExclusionAgent
from app.agents.decision import DecisionAgent
from app.agents.policy_evidence import PolicyEvidenceAgent
from app.agents.validation import ValidationAgent, force_review
from app.config import Settings, get_settings
from app.graph.state import ClaimWorkflowState
from app.models.schemas import (
    Decision,
    FindingStatus,
    Severity,
    ValidationStatus,
)
from app.retrieval.index import HybridRetriever

logger = logging.getLogger(__name__)

NODE_CASE_ANALYSIS = "case_analysis"
NODE_POLICY_EVIDENCE = "policy_evidence"
NODE_COVERAGE = "coverage_exclusion"
NODE_DECISION = "decision"
NODE_VALIDATION = "validation"


class ClaimWorkflow:
    """Builds and runs the five-agent graph."""

    def __init__(self, retriever: HybridRetriever, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.case_agent = CaseAnalysisAgent()
        self.evidence_agent = PolicyEvidenceAgent(retriever)
        self.coverage_agent = CoverageExclusionAgent()
        self.decision_agent = DecisionAgent(self.settings)
        self.validation_agent = ValidationAgent(self.settings)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def _node_case_analysis(self, state: ClaimWorkflowState) -> dict:
        case = state["case"]
        with AgentTimer(
            agent=NODE_CASE_ANALYSIS, action="Extracted claim facts and built investigation plan"
        ) as timer:
            analysis, queries = self.case_agent.run(case)
            timer.metrics = {
                "decision_dimensions": len(analysis.decision_dimensions),
                "investigation_questions": len(queries),
                "missing_fields": len(analysis.missing_fields),
                "irrelevant_attributes_ignored": len(analysis.irrelevant_attributes),
                "input_warnings": len(analysis.input_warnings),
            }
            timer.status = "OK"
        return {"analysis": analysis, "queries": queries, "trace": [timer.event()]}

    def _node_policy_evidence(self, state: ClaimWorkflowState) -> dict:
        queries = state.get("queries", [])
        with AgentTimer(
            agent=NODE_POLICY_EVIDENCE, action="Hybrid retrieval with fusion and reranking"
        ) as timer:
            evidence, stats = self.evidence_agent.run(queries)
            per_dim = self.evidence_agent.coverage_by_dimension(queries, evidence)
            uncovered = [d for d, n in per_dim.items() if n == 0]
            timer.metrics = {
                "queries_executed": stats.queries_executed,
                "dense_results": stats.dense_results,
                "bm25_results": stats.bm25_results,
                "fused_results": stats.fused_results,
                "reranked_results": stats.reranked_results,
                "dropped_below_threshold": stats.dropped_below_threshold,
                "dimensions_without_evidence": uncovered,
                "retrieval_ms": stats.elapsed_ms,
            }
            timer.status = "OK" if evidence else "NO_EVIDENCE"

        return {
            "evidence": evidence,
            "retrieval_stats": {**timer.metrics, "per_dimension": per_dim},
            "trace": [timer.event()],
        }

    def _node_coverage(self, state: ClaimWorkflowState) -> dict:
        with AgentTimer(
            agent=NODE_COVERAGE, action="Evaluated coverage, waiting periods, exclusions and limits"
        ) as timer:
            assessment = self.coverage_agent.run(
                state["case"], state["analysis"], state.get("evidence", [])
            )
            timer.metrics = {
                "findings": len(assessment.findings),
                "violations": sum(
                    1 for f in assessment.findings if f.status == FindingStatus.VIOLATED
                ),
                "unresolved": len(assessment.unresolved_dimensions),
                "limits_applied": len(assessment.applied_limits),
                "binding_limits": sum(1 for l in assessment.applied_limits if l.binding),
                "missing_evidence": len(assessment.missing_evidence),
                "total_deduction_inr": assessment.total_deduction_inr,
            }
            timer.status = "OK"
        return {"assessment": assessment, "trace": [timer.event()]}

    def _node_decision(self, state: ClaimWorkflowState) -> dict:
        revision = state.get("revision_count", 0)
        action = (
            "Generated structured decision"
            if revision == 0
            else f"Revised decision after validation failure (attempt {revision + 1})"
        )
        with AgentTimer(agent=NODE_DECISION, action=action) as timer:
            draft = self.decision_agent.run(
                state["case"],
                state["analysis"],
                state["assessment"],
                state.get("evidence", []),
                validation=state.get("validation"),
            )
            timer.metrics = {
                "decision": draft.decision.value,
                "confidence": draft.confidence,
                "citations": len(draft.citations),
                "key_findings": len(draft.key_findings),
                "payable_estimate_inr": draft.payable_estimate_inr,
            }
            timer.status = draft.decision.value
        return {"decision_draft": draft, "trace": [timer.event()]}

    def _node_validation(self, state: ClaimWorkflowState) -> dict:
        draft = state["decision_draft"]
        with AgentTimer(agent=NODE_VALIDATION, action="Verified statements against evidence") as timer:
            report = self.validation_agent.run(draft, state.get("evidence", []))
            report.revisions_applied = state.get("revision_count", 0)
            timer.metrics = {
                "checks_run": len(report.checks_run),
                "checks_failed": report.checks_failed,
                "unsupported_claims": len(report.unsupported_claims),
                "citation_integrity": report.citation_integrity,
            }
            timer.status = report.status.value

        draft.validation = report
        updates: dict = {"validation": report, "decision_draft": draft, "trace": [timer.event()]}

        if report.status == ValidationStatus.FAIL:
            revision = state.get("revision_count", 0)
            if revision >= self.settings.max_validation_revisions:
                logger.warning(
                    "Validation failed after %d revision(s) for %s; forcing NEEDS_REVIEW",
                    revision, draft.case_id,
                )
                # Clearing revision_required is what terminates the loop: the
                # decision has been forced to NEEDS_REVIEW, so there is nothing
                # further for the Decision Agent to revise.
                report.revision_required = False
                updates["decision_draft"] = force_review(draft, report)
            updates["revision_count"] = revision + 1
        return updates

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _route_after_validation(self, state: ClaimWorkflowState) -> str:
        report = state.get("validation")
        if report is None or report.status == ValidationStatus.PASS:
            return "end"
        if state.get("revision_count", 0) > self.settings.max_validation_revisions:
            return "end"
        return NODE_DECISION if report.revision_required else "end"

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("LangGraph unavailable (%s); using built-in runner", exc)
            return None

        graph = StateGraph(ClaimWorkflowState)
        graph.add_node(NODE_CASE_ANALYSIS, self._node_case_analysis)
        graph.add_node(NODE_POLICY_EVIDENCE, self._node_policy_evidence)
        graph.add_node(NODE_COVERAGE, self._node_coverage)
        graph.add_node(NODE_DECISION, self._node_decision)
        graph.add_node(NODE_VALIDATION, self._node_validation)

        graph.add_edge(START, NODE_CASE_ANALYSIS)
        graph.add_edge(NODE_CASE_ANALYSIS, NODE_POLICY_EVIDENCE)
        graph.add_edge(NODE_POLICY_EVIDENCE, NODE_COVERAGE)
        graph.add_edge(NODE_COVERAGE, NODE_DECISION)
        graph.add_edge(NODE_DECISION, NODE_VALIDATION)
        graph.add_conditional_edges(
            NODE_VALIDATION,
            self._route_after_validation,
            {NODE_DECISION: NODE_DECISION, "end": END},
        )
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _run_fallback(self, state: ClaimWorkflowState) -> ClaimWorkflowState:
        """Executes the identical graph without LangGraph."""
        def merge(base: ClaimWorkflowState, updates: dict) -> ClaimWorkflowState:
            for key, value in updates.items():
                if key in {"trace", "errors"}:
                    base[key] = [*base.get(key, []), *value]
                else:
                    base[key] = value
            return base

        state = merge(state, self._node_case_analysis(state))
        state = merge(state, self._node_policy_evidence(state))
        state = merge(state, self._node_coverage(state))

        for _ in range(self.settings.max_validation_revisions + 2):
            state = merge(state, self._node_decision(state))
            state = merge(state, self._node_validation(state))
            if self._route_after_validation(state) != NODE_DECISION:
                break
        return state

    def run(self, case) -> ClaimWorkflowState:
        initial: ClaimWorkflowState = {
            "case": case,
            "revision_count": 0,
            "trace": [],
            "errors": [],
        }
        if self._graph is None:
            return self._run_fallback(initial)
        # Allow room for the revision loop plus the five nodes.
        return self._graph.invoke(initial, {"recursion_limit": 25})

    @property
    def orchestrator(self) -> str:
        return "langgraph" if self._graph is not None else "builtin_state_machine"
