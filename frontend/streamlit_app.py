"""Reviewer-facing Streamlit frontend.

Built for an insurance reviewer who has to defend the decision, so the layout
follows the question they actually ask: *what was decided, on what clause, and
what is still missing?* Abstention is given the most prominent treatment on the
page, because the whole point of the system is that "I cannot safely decide" is
a first-class answer rather than a failure.

The app talks to the deployed FastAPI backend when ``API_BASE_URL`` is set and
otherwise calls the engine in-process, so it runs standalone on Streamlit
Community Cloud with or without a separate backend.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

try:  # same optional-dotenv convenience as app/config.py
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Aptino Claim Decision Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_CASES = PROJECT_ROOT / "evaluation" / "public_cases" / "public_test_cases.json"
CUSTOM_CASES_DIR = PROJECT_ROOT / "evaluation" / "custom_cases"
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
# A local LLM (e.g. ollama) is far slower than a hosted one and can exceed
# the default budget, especially on a cold model load.
ANALYZE_TIMEOUT_S = float(os.getenv("UI_ANALYZE_TIMEOUT_S", "120"))

DECISION_STYLE = {
    "ADMISSIBLE": ("#0f7b34", "✅", "Admissible"),
    "ADMISSIBLE_WITH_LIMITS": ("#8a6d00", "⚠️", "Admissible with limits"),
    "PARTIALLY_ADMISSIBLE": ("#a15c00", "◐", "Partially admissible"),
    "NOT_ADMISSIBLE": ("#a01b1b", "⛔", "Not admissible"),
    "NEEDS_REVIEW": ("#1f4e9c", "🔎", "Needs review — insufficient evidence"),
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    if PUBLIC_CASES.exists():
        for case in json.loads(PUBLIC_CASES.read_text(encoding="utf-8")):
            cases[f"{case['case_id']} (public)"] = case
    if CUSTOM_CASES_DIR.exists():
        for path in sorted(CUSTOM_CASES_DIR.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for case in payload if isinstance(payload, list) else [payload]:
                if isinstance(case, dict) and "case_id" in case:
                    cases[f"{case['case_id']} (custom)"] = case
    return cases


def analyze_case(case: dict[str, Any]) -> dict[str, Any]:
    """Call the deployed API, or the engine in-process when none is configured."""
    if API_BASE_URL:
        import httpx

        with httpx.Client(timeout=ANALYZE_TIMEOUT_S) as client:
            response = client.post(f"{API_BASE_URL}/analyze", json=case)
            response.raise_for_status()
            return response.json()

    from app.models.schemas import ClaimCase
    from app.services.engine import get_engine

    return get_engine().analyze(ClaimCase(**case)).model_dump(mode="json")


def backend_status() -> tuple[bool, str]:
    if API_BASE_URL:
        try:
            import httpx

            with httpx.Client(timeout=15.0) as client:
                data = client.get(f"{API_BASE_URL}/health").json()
            return data.get("index_ready", False), (
                f"Remote API · {data.get('chunks_indexed', 0)} chunks · v{data.get('version','?')}"
            )
        except Exception as exc:
            return False, f"Remote API unreachable: {type(exc).__name__}"
    try:
        from app.retrieval.index import get_retriever

        retriever = get_retriever()
        return retriever.ready, f"In-process engine · {len(retriever.chunks)} chunks"
    except Exception as exc:
        return False, f"Engine unavailable: {type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_decision_banner(result: dict) -> None:
    decision = result.get("decision", "NEEDS_REVIEW")
    colour, icon, label = DECISION_STYLE.get(decision, ("#444", "•", decision))
    confidence = result.get("confidence", 0.0)

    st.markdown(
        f"""
        <div style="border-left:8px solid {colour};background:rgba(128,128,128,0.09);
                    padding:1.1rem 1.3rem;border-radius:8px;margin-bottom:1rem;">
          <div style="font-size:1.55rem;font-weight:700;color:{colour};">
            {icon}&nbsp;{label}
          </div>
          <div style="opacity:0.85;margin-top:0.45rem;font-size:1.02rem;">
            {result.get("summary", "")}
          </div>
          <div style="margin-top:0.6rem;font-size:0.9rem;opacity:0.75;">
            Case <b>{result.get("case_id", "?")}</b> &nbsp;·&nbsp;
            Confidence <b>{confidence:.2f}</b> &nbsp;·&nbsp;
            Validation <b>{result.get("validation", {}).get("status", "?")}</b> &nbsp;·&nbsp;
            {result.get("total_elapsed_ms", 0)} ms
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if decision == "NEEDS_REVIEW":
        st.error(
            "**The system has abstained.** It did not reach a decision because the policy "
            "evidence does not support one safely.\n\n"
            f"**Reason:** {result.get('abstain_reason') or 'Insufficient evidence.'}",
            icon="🔎",
        )


def render_headline_metrics(result: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Confidence", f"{result.get('confidence', 0):.2f}")
    cols[1].metric("Claimed (INR)", f"{result.get('claimed_total_inr') or 0:,.0f}")
    cols[2].metric(
        "Deductions (INR)",
        f"{result.get('total_deduction_inr', 0):,.0f}",
        delta=None if not result.get("total_deduction_inr") else "capped by policy",
        delta_color="inverse",
    )
    payable = result.get("payable_estimate_inr")
    cols[3].metric("Payable estimate (INR)", "—" if payable is None else f"{payable:,.0f}")
    cols[4].metric("Citations", len(result.get("citations", [])))


def render_findings(result: dict) -> None:
    findings = result.get("key_findings", [])
    if not findings:
        st.info("No findings were produced.")
        return

    badge = {
        "VIOLATED": "🔴", "UNRESOLVED": "🔵", "LIMIT_APPLIES": "🟡",
        "SATISFIED": "🟢", "NOT_APPLICABLE": "⚪",
    }
    material = [f for f in findings if f["severity"] in {"BLOCKING", "ABSTAIN", "LIMITING"}]
    supporting = [f for f in findings if f not in material]

    if material:
        st.markdown("**Decisive findings**")
        for finding in material:
            _render_finding(finding, badge, expanded=True)
    if supporting:
        st.markdown("**Supporting findings**")
        for finding in supporting:
            _render_finding(finding, badge, expanded=False)


def _render_finding(finding: dict, badge: dict, *, expanded: bool) -> None:
    icon = badge.get(finding["status"], "•")
    title = f"{icon} [{finding['severity']}] {finding['statement'][:120]}"
    with st.expander(title, expanded=expanded):
        st.write(finding["statement"])
        if finding.get("detail"):
            st.caption(finding["detail"])
        st.caption(
            f"Rule `{finding['rule_id']}` · dimension `{finding['dimension']}` · "
            f"status `{finding['status']}`"
        )
        for citation in finding.get("citations", []):
            st.markdown(
                f"> **{citation['source']}** — p.{citation['page']} · "
                f"*{citation['section']}* · `{citation['chunk_id']}`"
            )
            if citation.get("quote"):
                st.markdown(
                    f"<div style='border-left:3px solid #888;padding-left:0.8rem;"
                    f"opacity:0.85;font-size:0.9rem;'>{citation['quote']}</div>",
                    unsafe_allow_html=True,
                )
        if not finding.get("citations"):
            st.caption("_No citation — this finding reports that the clause could not be established._")


def render_limits(result: dict) -> None:
    limits = result.get("applicable_limits", [])
    if not limits:
        st.info("No monetary limits were applied to this claim.")
        return

    st.dataframe(
        [
            {
                "Category": l["category"],
                "Cap (INR)": l.get("limit_amount_inr"),
                "Claimed (INR)": l.get("claimed_amount_inr"),
                "Allowed (INR)": l.get("allowed_amount_inr"),
                "Deduction (INR)": l.get("deduction_inr", 0),
                "Binding": "Yes" if l.get("binding") else "No",
            }
            for l in limits
        ],
        use_container_width=True,
        hide_index=True,
    )
    for limit in limits:
        marker = "**binding**" if limit.get("binding") else "not binding"
        with st.expander(f"{limit['category']} — {marker}"):
            st.write(limit["description"])
            st.caption(f"Basis: {limit['basis']}")
            for citation in limit.get("citations", []):
                st.markdown(
                    f"> p.{citation['page']} · *{citation['section']}* · `{citation['chunk_id']}`"
                )


def render_missing_evidence(result: dict) -> None:
    missing = result.get("missing_evidence", [])
    if not missing:
        st.success("No evidence gaps were identified.")
        return
    for item in missing:
        tag = "🚫 Blocking" if item.get("blocking") else "ℹ️ Non-blocking"
        st.markdown(f"**{tag} — {item['item']}**")
        st.caption(item["why_required"])
        if item.get("suggested_document"):
            st.caption(f"Suggested document: `{item['suggested_document']}`")
        st.divider()


def render_evidence(result: dict) -> None:
    evidence = result.get("evidence", [])
    if not evidence:
        st.info("No evidence was returned for this request.")
        return

    cited = {c["chunk_id"] for c in result.get("citations", [])}
    st.caption(
        f"{len(evidence)} chunks retrieved · {len(cited)} cited in the decision. "
        "Scores: retrieval = fused rank score, rerank = cross-encoder stage."
    )
    only_cited = st.checkbox("Show only chunks cited in the decision", value=False)

    for item in evidence:
        is_cited = item["chunk_id"] in cited
        if only_cited and not is_cited:
            continue
        marker = "📌 CITED" if is_cited else "· retrieved"
        header = (
            f"{marker} | p.{item['page']} | {item['section']} — {item['heading'][:60]} "
            f"| rerank {item['rerank_score']:.3f}"
        )
        with st.expander(header, expanded=False):
            cols = st.columns(4)
            cols[0].metric("Rerank", f"{item['rerank_score']:.3f}")
            cols[1].metric("Fusion", f"{item['retrieval_score']:.4f}")
            cols[2].metric(
                "Dense",
                "—" if item.get("dense_score") is None else f"{item['dense_score']:.3f}",
                help=f"rank {item.get('dense_rank')}" if item.get("dense_rank") else None,
            )
            cols[3].metric(
                "BM25",
                "—" if item.get("bm25_score") is None else f"{item['bm25_score']:.2f}",
                help=f"rank {item.get('bm25_rank')}" if item.get("bm25_rank") else None,
            )
            st.caption(
                f"`{item['chunk_id']}` · method **{item['retrieval_method']}** · "
                f"dimensions: {', '.join(item.get('matched_dimensions', [])) or '—'}"
            )
            st.markdown(
                f"<div style='background:rgba(128,128,128,0.08);padding:0.8rem;"
                f"border-radius:6px;font-size:0.9rem;'>{item['text']}</div>",
                unsafe_allow_html=True,
            )


def render_trace(result: dict) -> None:
    trace = result.get("trace", [])
    if not trace:
        st.info("No trace was returned.")
        return

    st.caption(
        "Auditable execution record: agent actions, retrieval counts, validation status and "
        "timings. Internal model reasoning is deliberately not exposed."
    )
    st.dataframe(
        [
            {
                "Agent": e["agent"],
                "Action": e["action"],
                "Status": e.get("status") or "—",
                "Elapsed (ms)": e.get("elapsed_ms", 0),
            }
            for e in trace
        ],
        use_container_width=True,
        hide_index=True,
    )
    for event in trace:
        if event.get("metrics"):
            with st.expander(f"{event['agent']} — metrics"):
                st.json(event["metrics"])

    st.markdown("**Confidence breakdown**")
    breakdown = result.get("confidence_breakdown", {})
    if breakdown:
        cols = st.columns(4)
        cols[0].metric("Retrieval quality", f"{breakdown.get('retrieval_quality', 0):.2f}")
        cols[1].metric("Evidence coverage", f"{breakdown.get('evidence_coverage', 0):.2f}")
        cols[2].metric("Citation support", f"{breakdown.get('citation_support', 0):.2f}")
        cols[3].metric("Decision consistency", f"{breakdown.get('decision_consistency', 0):.2f}")
        st.caption(
            f"raw {breakdown.get('raw_score', 0):.3f} × validation "
            f"{breakdown.get('validation_factor', 1):.2f} − penalty "
            f"{breakdown.get('missing_evidence_penalty', 0):.3f} = "
            f"**{breakdown.get('final_score', 0):.3f}**"
        )
        for note in breakdown.get("notes", []):
            st.caption(f"• {note}")


def render_validation(result: dict) -> None:
    validation = result.get("validation", {})
    passed = validation.get("status") == "PASS"
    (st.success if passed else st.warning)(
        f"Validation **{validation.get('status', '?')}** — "
        f"{len(validation.get('checks_run', []))} checks run, "
        f"{len(validation.get('checks_failed', []))} failed, "
        f"{validation.get('revisions_applied', 0)} revision(s) applied."
    )
    if validation.get("checks_run"):
        st.caption("Checks run: " + ", ".join(f"`{c}`" for c in validation["checks_run"]))
    if validation.get("checks_failed"):
        st.caption("Failed: " + ", ".join(f"`{c}`" for c in validation["checks_failed"]))
    for claim in validation.get("unsupported_claims", []):
        st.markdown(f"- ⚠️ {claim}")
    for note in validation.get("notes", []):
        st.caption(f"• {note}")


def render_investigation(result: dict) -> None:
    analysis = result.get("case_analysis") or {}
    queries = result.get("retrieval_queries", [])

    cols = st.columns(3)
    cols[0].metric("Decision dimensions", len(analysis.get("decision_dimensions", [])))
    cols[1].metric("Retrieval queries", len(queries))
    cols[2].metric("Missing input fields", len(analysis.get("missing_fields", [])))

    if analysis.get("input_warnings"):
        for warning in analysis["input_warnings"]:
            st.warning(warning)

    st.markdown("**Investigation plan** — one targeted query per decision dimension")
    for item in analysis.get("investigation_plan", []):
        flag = " 🔒 blocking if unresolved" if item.get("blocking_if_unresolved") else ""
        with st.expander(f"`{item['dimension']}`{flag}"):
            st.write(item["why_it_matters"])
            st.caption(f"Query: {item['question']}")
            if item.get("required_inputs"):
                st.caption("Required inputs: " + ", ".join(f"`{r}`" for r in item["required_inputs"]))

    if analysis.get("missing_fields"):
        st.markdown("**Missing input fields**")
        st.caption(", ".join(f"`{f}`" for f in analysis["missing_fields"]))

    if analysis.get("irrelevant_attributes"):
        st.markdown("**Attributes considered and discounted as policy-irrelevant**")
        for attr in analysis["irrelevant_attributes"]:
            st.caption(f"• {attr}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.title("⚖️ Policy-Aware Multi-Agent RAG Claim Decision Engine")
    st.caption(
        "Health-insurance claim adjudication grounded in the supplied policy wording. "
        "Every material statement is bound to a retrieved clause; where the policy does not "
        "support a safe conclusion the engine abstains."
    )

    ready, status_text = backend_status()
    (st.sidebar.success if ready else st.sidebar.error)(status_text)

    st.sidebar.header("Claim input")
    mode = st.sidebar.radio(
        "Source", ["Select a case", "Paste JSON", "Upload JSON"], label_visibility="collapsed"
    )

    cases = load_cases()
    case: dict | None = None

    if mode == "Select a case":
        if not cases:
            st.sidebar.warning("No case files found.")
        else:
            key = st.sidebar.selectbox("Case", list(cases.keys()))
            case = cases[key]
    elif mode == "Paste JSON":
        text = st.sidebar.text_area("Claim case JSON", height=280, placeholder='{"case_id": "..."}')
        if text.strip():
            try:
                case = json.loads(text)
            except json.JSONDecodeError as exc:
                st.sidebar.error(f"Invalid JSON: {exc}")
    else:
        upload = st.sidebar.file_uploader("Claim case JSON", type=["json"])
        if upload is not None:
            try:
                payload = json.loads(upload.read().decode("utf-8"))
                case = payload[0] if isinstance(payload, list) and payload else payload
            except Exception as exc:
                st.sidebar.error(f"Could not read file: {exc}")

    analyze_clicked = st.sidebar.button(
        "Analyze claim", type="primary", use_container_width=True, disabled=case is None
    )

    if case:
        with st.sidebar.expander("Case payload"):
            st.json(case)

    if analyze_clicked and case:
        with st.spinner("Running multi-agent analysis…"):
            try:
                st.session_state["result"] = analyze_case(case)
            except Exception as exc:
                st.session_state["result"] = None
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("result")
    if not result:
        st.info("Select or paste a claim case in the sidebar, then choose **Analyze claim**.")
        with st.expander("What this system does"):
            st.markdown(
                """
                Five specialised agents exchange structured state:

                1. **Case Analysis** — extracts facts, picks the decision dimensions this claim
                   turns on, and builds one targeted retrieval query per dimension.
                2. **Policy Evidence** — hybrid retrieval (dense + BM25 → RRF fusion → reranking)
                   over the policy wording, returning evidence with page/section/chunk metadata.
                3. **Coverage & Exclusion** — evaluates coverage, definitions, waiting periods,
                   exclusions and monetary limits. A rule fires only if its clause was retrieved.
                4. **Decision** — combines findings into one of five statuses and computes a
                   confidence score from measured evidential support.
                5. **Validation** — verifies every material statement against the retrieved
                   evidence and can force a revision or an abstention.
                """
            )
        return

    render_decision_banner(result)
    render_headline_metrics(result)

    tabs = st.tabs(
        ["Findings", "Limits & deductions", "Missing evidence", "Policy evidence",
         "Execution trace", "Validation", "Investigation", "Raw JSON"]
    )
    with tabs[0]:
        render_findings(result)
    with tabs[1]:
        render_limits(result)
    with tabs[2]:
        render_missing_evidence(result)
    with tabs[3]:
        render_evidence(result)
    with tabs[4]:
        render_trace(result)
    with tabs[5]:
        render_validation(result)
    with tabs[6]:
        render_investigation(result)
    with tabs[7]:
        st.download_button(
            "Download decision JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{result.get('case_id', 'decision')}_decision.json",
            mime="application/json",
        )
        st.json(result)


if __name__ == "__main__":
    main()
