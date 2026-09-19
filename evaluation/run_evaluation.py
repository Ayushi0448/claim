"""End-to-end evaluation harness.

    python -m evaluation.run_evaluation

Runs every public and custom case through the same engine the API uses, scores
the output against the hand-derived ground truth, and writes:

    evaluation/results/results.json     full per-case detail and aggregates
    evaluation/results/results.csv      one row per case
    evaluation/results/summary.md       human-readable report

Options:
    --suite {all,public,custom}   restrict the run
    --case CASE_ID                run a single case (repeatable)
    --output-dir PATH             write elsewhere
    --quiet                       suppress the per-case console table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import AnalysisResponse, ClaimCase  # noqa: E402
from app.services.engine import get_engine  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    CaseResult,
    aggregate,
    citation_correctness,
    citation_hit_rate,
    evidence_recall_at_k,
)

logger = logging.getLogger("evaluation")

PUBLIC_CASES = PROJECT_ROOT / "evaluation" / "public_cases" / "public_test_cases.json"
CUSTOM_CASES_DIR = PROJECT_ROOT / "evaluation" / "custom_cases"
EXPECTED_PUBLIC = PROJECT_ROOT / "evaluation" / "expected_results" / "expected_public.json"
EXPECTED_CUSTOM = PROJECT_ROOT / "evaluation" / "expected_results" / "expected_custom.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results"

# Deductions are compared with a small tolerance to absorb float rounding, not
# to hide disagreement: anything above this is reported as a mismatch.
DEDUCTION_TOLERANCE_INR = 1.0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_public_cases() -> list[dict]:
    if not PUBLIC_CASES.exists():
        raise FileNotFoundError(f"Supplied public cases not found at {PUBLIC_CASES}")
    return json.loads(PUBLIC_CASES.read_text(encoding="utf-8"))


def load_custom_cases() -> list[dict]:
    cases: list[dict] = []
    if not CUSTOM_CASES_DIR.exists():
        return cases
    for path in sorted(CUSTOM_CASES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(payload if isinstance(payload, list) else [payload])
    return cases


def load_expected(path: Path) -> dict[str, dict]:
    if not path.exists():
        logger.warning("Expected results file missing: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("cases", {})


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_case(
    response: AnalysisResponse, expected: dict | None, suite: str
) -> CaseResult:
    expected = expected or {}
    expected_decision = expected.get("expected_decision")
    required_evidence = expected.get("required_evidence", [])

    recall, found, missing = evidence_recall_at_k(response, required_evidence)
    hit_rate = citation_hit_rate(response, required_evidence)
    correctness = citation_correctness(response)

    expected_deduction = expected.get("expected_deduction_inr")
    actual_deduction = response.total_deduction_inr
    if expected_deduction is None:
        deduction_correct = None
    else:
        deduction_correct = abs(actual_deduction - expected_deduction) <= DEDUCTION_TOLERANCE_INR

    actual_decision = response.decision.value
    decision_correct = (
        None if expected_decision is None else actual_decision == expected_decision
    )

    notes: list[str] = []
    if decision_correct is False:
        notes.append(f"Expected {expected_decision}, produced {actual_decision}.")
    if deduction_correct is False:
        notes.append(
            f"Deduction mismatch: expected INR {expected_deduction:,.0f}, "
            f"produced INR {actual_deduction:,.0f}."
        )
    if missing:
        notes.append(f"{len(missing)} required clause(s) not retrieved.")

    return CaseResult(
        case_id=response.case_id,
        suite=suite,
        expected_decision=expected_decision,
        actual_decision=actual_decision,
        decision_correct=decision_correct,
        confidence=response.confidence,
        validation_status=response.validation.status.value,
        expected_deduction_inr=expected_deduction,
        actual_deduction_inr=actual_deduction,
        deduction_correct=deduction_correct,
        n_evidence=len(response.evidence),
        n_citations=len(response.citations),
        evidence_recall=recall,
        evidence_found=found,
        evidence_missing=missing,
        citation_hit_rate=hit_rate,
        citation_correctness=correctness,
        unsupported_claims=len(response.validation.unsupported_claims),
        latency_ms=response.total_elapsed_ms,
        abstained=response.abstained,
        expected_abstention=(expected_decision == "NEEDS_REVIEW"),
        notes=notes,
    )


def run_suite(
    cases: list[dict],
    expected_map: dict[str, dict],
    suite: str,
    *,
    quiet: bool = False,
) -> tuple[list[CaseResult], list[AnalysisResponse]]:
    engine = get_engine()
    results: list[CaseResult] = []
    responses: list[AnalysisResponse] = []

    for raw in cases:
        case_id = raw.get("case_id", "<unknown>")
        try:
            case = ClaimCase(**raw)
        except Exception as exc:
            logger.error("Skipping %s: invalid case (%s)", case_id, exc)
            continue

        response = engine.analyze(case)
        responses.append(response)
        result = score_case(response, expected_map.get(case_id), suite)
        results.append(result)

        if not quiet:
            mark = {True: "PASS", False: "FAIL", None: " -- "}[result.decision_correct]
            print(
                f"  [{mark}] {result.case_id:9s} {result.actual_decision:24s} "
                f"conf={result.confidence:.2f} val={result.validation_status:4s} "
                f"recall={_fmt(result.evidence_recall)} "
                f"cite_ok={_fmt(result.citation_correctness)} "
                f"{result.latency_ms:5d}ms"
            )
            for note in result.notes:
                print(f"           ! {note}")

    return results, responses


def _fmt(value: float | None) -> str:
    return " n/a" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_csv(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def build_summary(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines: list[str] = [
        "# Evaluation Results",
        "",
        f"_Generated {report['generated_at']} · engine v{report['engine_version']}_",
        "",
        "## Configuration",
        "",
        "| Component | Backend |",
        "| --- | --- |",
    ]
    for key, value in report["backends"].items():
        lines.append(f"| {key} | `{value}` |")

    lines += [
        "",
        "## Headline metrics (all cases)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases evaluated | {overall['n_cases']} ({overall['n_labelled']} labelled) |",
        f"| **Decision accuracy** | **{_pct(overall['decision_accuracy'])}** |",
        f"| Deduction accuracy | {_pct(overall['deduction_accuracy'])} "
        f"({overall['deduction_cases_checked']} cases with a monetary label) |",
        f"| Evidence recall@k | {_pct(overall['evidence_recall_at_k'])} |",
        f"| Citation hit rate | {_pct(overall['citation_hit_rate'])} |",
        f"| Citation correctness | {_pct(overall['citation_correctness'])} |",
        f"| Validation pass rate | {_pct(overall['validation_pass_rate'])} |",
        f"| Unsupported claims detected | {overall['unsupported_claims_total']} |",
        f"| Mean latency | {overall['latency_ms']['mean']} ms "
        f"(median {overall['latency_ms']['median']} ms, max {overall['latency_ms']['max']} ms) |",
        "",
        "## Abstention performance",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    ab = overall["abstention"]
    lines += [
        f"| Cases where NEEDS_REVIEW is correct | {ab['expected_abstentions']} |",
        f"| Abstentions produced | {ab['actual_abstentions']} |",
        f"| Correct abstentions | {ab['true_positives']} |",
        f"| Missed abstentions (decided when it should not have) | {ab['false_negatives']} |",
        f"| Over-abstentions (abstained when a decision was available) | {ab['false_positives']} |",
        f"| Precision / Recall / F1 | {_pct(ab['precision'])} / {_pct(ab['recall'])} / {_pct(ab['f1'])} |",
        "",
        "## Confidence calibration",
        "",
        f"- Mean confidence overall: **{overall['confidence']['mean']}**",
        f"- Mean confidence when the decision was correct: **{overall['confidence']['mean_when_correct']}**",
        f"- Mean confidence when the decision was incorrect: "
        f"**{overall['confidence']['mean_when_incorrect'] if overall['confidence']['mean_when_incorrect'] is not None else 'n/a (no incorrect decisions)'}**",
        "",
        "## By suite",
        "",
        "| Suite | Cases | Decision accuracy | Evidence recall@k | Citation correctness | Validation pass |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for suite_name, suite_report in report["suites"].items():
        lines.append(
            f"| {suite_name} | {suite_report['n_cases']} | "
            f"{_pct(suite_report['decision_accuracy'])} | "
            f"{_pct(suite_report['evidence_recall_at_k'])} | "
            f"{_pct(suite_report['citation_correctness'])} | "
            f"{_pct(suite_report['validation_pass_rate'])} |"
        )

    lines += [
        "",
        "## Accuracy by expected decision",
        "",
        "| Expected decision | Cases | Correct | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for decision, stats in sorted(overall["decision_accuracy_by_expected"].items()):
        accuracy = stats["correct"] / stats["expected"] if stats["expected"] else 0
        lines.append(
            f"| {decision} | {stats['expected']} | {stats['correct']} | {_pct(accuracy)} |"
        )

    lines += ["", "## Per-case results", "",
              "| Case | Suite | Expected | Actual | OK | Conf | Recall@k | Cite ✓ | Val | ms |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["cases"]:
        mark = {True: "✅", False: "❌", "": "—"}.get(row["decision_correct"], "—")
        lines.append(
            f"| {row['case_id']} | {row['suite']} | {row['expected_decision'] or '—'} | "
            f"{row['actual_decision']} | {mark} | {row['confidence']:.2f} | "
            f"{row['evidence_recall'] if row['evidence_recall'] != '' else '—'} | "
            f"{row['citation_correctness'] if row['citation_correctness'] != '' else '—'} | "
            f"{row['validation']} | {row['latency_ms']} |"
        )

    failures = [r for r in report["cases"] if r["decision_correct"] is False]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("No decision mismatches against the ground truth in this run.")
    else:
        for row in failures:
            lines.append(
                f"- **{row['case_id']}** — expected `{row['expected_decision']}`, "
                f"produced `{row['actual_decision']}` "
                f"(confidence {row['confidence']:.2f}, validation {row['validation']})."
            )
            if row.get("evidence_missing"):
                lines.append(f"  - Required clauses not retrieved: {row['evidence_missing']}")

    gaps = [r for r in report["cases"] if r.get("evidence_missing")]
    lines += ["", "## Retrieval gaps", ""]
    if not gaps:
        lines.append("Every required policy clause was retrieved for every case.")
    else:
        for row in gaps:
            lines.append(f"- **{row['case_id']}**: {row['evidence_missing']}")

    lines += [
        "",
        "---",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "python -m evaluation.run_evaluation",
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the claim-engine evaluation suite.")
    parser.add_argument("--suite", choices=["all", "public", "custom"], default="all")
    parser.add_argument("--case", action="append", default=[], help="Run only this case id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))

    started = time.perf_counter()
    engine = get_engine()
    backends = engine.backends()

    print("=" * 96)
    print("Aptino Policy-Aware Multi-Agent RAG Claim Decision Engine — Evaluation")
    print("=" * 96)
    print("Backends: " + ", ".join(f"{k}={v}" for k, v in backends.items()))
    print()

    all_results: list[CaseResult] = []
    suite_reports: dict[str, Any] = {}
    responses_by_suite: dict[str, list[AnalysisResponse]] = {}

    def wanted(cases: list[dict]) -> list[dict]:
        if not args.case:
            return cases
        return [c for c in cases if c.get("case_id") in set(args.case)]

    if args.suite in {"all", "public"}:
        cases = wanted(load_public_cases())
        if cases:
            print(f"PUBLIC SUITE ({len(cases)} supplied cases, unmodified)")
            results, responses = run_suite(
                cases, load_expected(EXPECTED_PUBLIC), "public", quiet=args.quiet
            )
            all_results += results
            responses_by_suite["public"] = responses
            suite_reports["public"] = aggregate(results)
            print()

    if args.suite in {"all", "custom"}:
        cases = wanted(load_custom_cases())
        if cases:
            print(f"CUSTOM SUITE ({len(cases)} candidate-authored cases)")
            results, responses = run_suite(
                cases, load_expected(EXPECTED_CUSTOM), "custom", quiet=args.quiet
            )
            all_results += results
            responses_by_suite["custom"] = responses
            suite_reports["custom"] = aggregate(results)
            print()

    if not all_results:
        print("No cases were evaluated.")
        return 1

    overall = aggregate(all_results)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": backends.get("engine_version", "unknown"),
        "backends": backends,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "overall": overall,
        "suites": suite_reports,
        "cases": [r.to_row() for r in all_results],
        "case_detail": [
            {
                "case_id": r.case_id,
                "suite": r.suite,
                "expected_decision": r.expected_decision,
                "actual_decision": r.actual_decision,
                "decision_correct": r.decision_correct,
                "evidence_found": r.evidence_found,
                "evidence_missing": r.evidence_missing,
                "notes": r.notes,
            }
            for r in all_results
        ],
    }

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(all_results, output_dir / "results.csv")
    summary = build_summary(report)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    # Full decision payloads, so a reviewer can inspect any case offline.
    decisions = {
        suite: [r.model_dump(mode="json") for r in responses]
        for suite, responses in responses_by_suite.items()
    }
    (output_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"  Cases evaluated      : {overall['n_cases']} ({overall['n_labelled']} labelled)")
    print(f"  Decision accuracy    : {_pct(overall['decision_accuracy'])}")
    print(f"  Deduction accuracy   : {_pct(overall['deduction_accuracy'])} "
          f"({overall['deduction_cases_checked']} cases)")
    print(f"  Evidence recall@k    : {_pct(overall['evidence_recall_at_k'])}")
    print(f"  Citation hit rate    : {_pct(overall['citation_hit_rate'])}")
    print(f"  Citation correctness : {_pct(overall['citation_correctness'])}")
    print(f"  Validation pass rate : {_pct(overall['validation_pass_rate'])}")
    ab = overall["abstention"]
    print(f"  Abstention P/R/F1    : {_pct(ab['precision'])} / {_pct(ab['recall'])} / {_pct(ab['f1'])}"
          f"  (TP={ab['true_positives']} FP={ab['false_positives']} FN={ab['false_negatives']})")
    print(f"  Mean latency         : {overall['latency_ms']['mean']} ms")
    print()
    print(f"  Wrote {output_dir/'results.json'}")
    print(f"  Wrote {output_dir/'results.csv'}")
    print(f"  Wrote {output_dir/'summary.md'}")
    print(f"  Wrote {output_dir/'decisions.json'}")

    failures = [r for r in all_results if r.decision_correct is False]
    if failures:
        print()
        print(f"  {len(failures)} FAILING CASE(S): " + ", ".join(r.case_id for r in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
