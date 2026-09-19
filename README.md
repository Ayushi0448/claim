# Aptino — Policy-Aware Multi-Agent RAG Claim Decision Engine

A production-style system that adjudicates health-insurance claims against the
supplied policy wording (Universal Sompo CSC Individual Health Insurance,
UNIHLIP18004V011718), using hybrid retrieval and a five-agent workflow.

**Every material statement is bound to a clause that retrieval actually
returned.** Where the policy does not support a safe conclusion, the system
returns `NEEDS_REVIEW` instead of guessing.

| Metric | Result |
| --- | --- |
| Decision accuracy | **100%** (22/22 cases) |
| Deduction accuracy | **100%** (11 cases with a monetary label) |
| Evidence recall@k | **100%** |
| Citation correctness | **100%** |
| Abstention precision / recall / F1 | **100% / 100% / 100%** |
| Validation pass rate | **100%** |
| Mean latency | **264 ms** per claim (hardware-dependent; see §25) |
| Tests | **152 passing** |

Reproduce with `python -m evaluation.run_evaluation`. Full report:
[`evaluation/results/summary.md`](evaluation/results/summary.md).
These numbers come with real caveats — see [§28 Known limitations](#28-known-limitations).

---

## Table of contents

1. [Problem statement](#1-problem-statement) · 2. [Architecture](#2-architecture) ·
3. [Architecture diagram](#3-architecture-diagram) · 4. [Agent responsibilities](#4-agent-responsibilities) ·
5. [LangGraph state flow](#5-langgraph-state-flow) · 6. [RAG pipeline](#6-rag-pipeline) ·
7. [Chunking strategy](#7-chunking-strategy) · 8. [Dense retrieval](#8-dense-retrieval) ·
9. [BM25 retrieval](#9-bm25-retrieval) · 10. [Fusion](#10-fusion) · 11. [Reranking](#11-reranking) ·
12. [Citation design](#12-citation-design) · 13. [Abstention strategy](#13-abstention-strategy) ·
14. [Confidence calculation](#14-confidence-calculation) · 15. [API documentation](#15-api-documentation) ·
16. [Frontend](#16-frontend) · 17. [Local setup](#17-local-setup) ·
18. [Environment variables](#18-environment-variables) · 19. [Running ingestion](#19-running-ingestion) ·
20. [Running the API](#20-running-the-api) · 21. [Running the frontend](#21-running-the-frontend) ·
22. [Running the evaluation](#22-running-the-evaluation) · 23. [Test commands](#23-test-commands) ·
24. [Deployment](#24-deployment) · 25. [Evaluation results](#25-evaluation-results) ·
26. [Failure analysis](#26-failure-analysis) · 27. [Design trade-offs](#27-design-trade-offs) ·
28. [Known limitations](#28-known-limitations) · 29. [Security considerations](#29-security-considerations)

---

## 1. Problem statement

Given a synthetic health-insurance claim, decide whether it is payable under the
supplied policy — and justify the decision clause by clause.

The hard parts are not generative:

- The governing clause is buried in a 17-page legal document.
- A decision often requires combining clauses from different sections (an
  exclusion *and* its continuity waiver; a definition *and* an exclusion).
- A waiting period can defeat a claim whose treatment is otherwise covered.
- A category sub-limit changes the payable amount without affecting admissibility.
- Some claims genuinely cannot be decided on the evidence supplied.

Constraints taken as binding: the supplied policy is the **only** source of
policy rules; external insurance knowledge may not be used to invent a
conclusion; where evidence is insufficient the system must abstain; the supplied
public cases must not be modified.

---

## 2. Architecture

The system splits the problem along the line between what is genuinely hard and
what is merely arithmetic:

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| **Retrieval** | Find the governing clauses | Hybrid dense + BM25 → RRF → rerank |
| **Reasoning** | Apply them to the facts | Deterministic rule engine, evidence-bound |
| **Verification** | Confirm the output is supported | Independent validation agent |
| **Language** | Phrase the result | Optional LLM, cannot change a decision |

### The core mechanism: evidence binding

Every policy rule declares the **anchor text** it depends on — a verbatim
fragment of the clause that gives it authority. A rule may only fire if that
anchor appears in a chunk **retrieval actually returned for this case**.

```python
RULE_PED = PolicyRule(
    rule_id="pre_existing_disease_48m",
    anchors=[Anchor(phrase="Pre-existing diseases will not be covered until 48 months")],
    applies=lambda ctx: ctx.case.treatment.pre_existing is True,
    evaluate=_eval_ped,
)
```

This makes three of the assignment's reliability rules structural rather than
aspirational:

- **Never invent a clause** — the anchor would not resolve.
- **Never fabricate a citation** — page, section and `chunk_id` are copied from
  the resolved chunk; no code path constructs them.
- **Abstain when evidence is missing** — an unresolved anchor yields
  `UNRESOLVED`, which propagates to `NEEDS_REVIEW`.

**The failure mode this buys:** a retrieval regression degrades the system into
abstention, never into confident wrong answers. That is not theoretical — it is
exactly what happened during development (failure F-04), where a bad query
change made five cases abstain rather than answer incorrectly.

### Why the LLM is optional

Whether 27 months clears a 48-month waiting period is arithmetic. Encoding it as
rules gives exact computation, reproducible evaluation, and an auditable link
from every sentence to the clause behind it. The LLM polishes prose, proposes
extra retrieval phrasings, and gives an advisory second opinion during
validation — **decisions are identical with `LLM_PROVIDER=none`**, which is what
makes the metrics above reproducible and the system free to run.

---

## 3. Architecture diagram

```
                              ClaimCase (JSON)
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │  AGENT 1 · Case Analysis                 │
              │  facts · derived facts · dimensions      │
              │  missing fields · irrelevant attributes  │
              │  investigation plan                      │
              └───────────────────┬──────────────────────┘
                                  │ CaseAnalysis + list[RetrievalQuery]
                                  │  (topical queries + anchor-directed queries)
                                  ▼
              ┌──────────────────────────────────────────┐
              │  AGENT 2 · Policy Evidence               │
              │                                          │
              │   ┌─────────────┐      ┌──────────────┐  │
              │   │ Dense       │      │ BM25         │  │
              │   │ (cosine)    │      │ (Okapi)      │  │
              │   └──────┬──────┘      └──────┬───────┘  │
              │          └───────┬────────────┘          │
              │                  ▼                       │
              │        Reciprocal Rank Fusion            │
              │                  ▼                       │
              │             Reranking                    │
              │                  ▼                       │
              │   Phase 1: per-dimension recall floor    │
              │   Phase 2: cross-dimension precision fill│
              └───────────────────┬──────────────────────┘
                                  │ list[EvidenceItem]
                                  │  (page · section · chunk_id · all scores)
                                  ▼
              ┌──────────────────────────────────────────┐
              │  AGENT 3 · Coverage & Exclusion          │
              │  ┌────────────────────────────────────┐  │
              │  │ EVIDENCE BINDING                   │  │
              │  │ anchor resolved? ──no──→ UNRESOLVED│  │
              │  │        │yes                        │  │
              │  │        ▼                           │  │
              │  │ rule fires → Finding + Citation    │  │
              │  └────────────────────────────────────┘  │
              │  + limit calculators (room/ICU/fees/…)   │
              └───────────────────┬──────────────────────┘
                                  │ CoverageAssessment
                                  ▼
              ┌──────────────────────────────────────────┐
              │  AGENT 4 · Decision                      │
              │  blocking → NOT_ADMISSIBLE               │
              │  unresolved → NEEDS_REVIEW               │
              │  partial → PARTIALLY_ADMISSIBLE          │
              │  binding cap → ADMISSIBLE_WITH_LIMITS    │
              │  else → ADMISSIBLE                       │
              │  + measured confidence                   │
              └───────────────────┬──────────────────────┘
                                  │ AnalysisResponse (draft)
                                  ▼
              ┌──────────────────────────────────────────┐
              │  AGENT 5 · Validation                    │
              │  7 checks vs retrieved evidence          │
              │  FAIL → revise ──┐                       │
              │  FAIL again → force NEEDS_REVIEW         │
              └───────────────────┬──────────────────────┘
                     ▲            │ PASS
                     └── revise ──┤
                                  ▼
                 Structured decision + citations + trace
```

---

## 4. Agent responsibilities

| # | Agent | Owns | Produces |
| --- | --- | --- | --- |
| 1 | **Case Analysis** | Turning a claim into an investigation | `CaseAnalysis`, `list[RetrievalQuery]` |
| 2 | **Policy Evidence** | The retrieval stack | `list[EvidenceItem]`, `RetrievalStats` |
| 3 | **Coverage & Exclusion** | Interpreting clauses against facts | `CoverageAssessment` |
| 4 | **Decision** | Precedence and confidence | `AnalysisResponse` (draft) |
| 5 | **Validation** | Verifying support | `ValidationReport` |

**What makes the separation real, not decorative:**

- Agent 2 performs **no interpretation**. This is what lets Agent 5 check
  Agent 3's conclusions against the evidence — if one component both retrieved
  and interpreted, validation would be grading its own work.
- Agent 3 reaches **no verdict**. It reports what the policy says; Agent 4
  decides. That keeps the precedence logic auditable in isolation.
- Agent 1's output is **claim-driven**: a domiciliary claim pulls in the
  domiciliary dimension and its sub-limit; an 8-hour admission pulls in day care.
- Agent 5 re-derives support **independently** and can force the decision to
  change.

Each agent is unit-tested alone, including the critical negative case:
`test_rule_cannot_fire_without_its_clause` runs Agent 3 with `evidence=[]` and
asserts it yields `UNRESOLVED` findings with zero citations, rather than falling
back on built-in knowledge.

---

## 5. LangGraph state flow

Agents exchange typed Pydantic models — never free-form prose. Ownership is
declared in `app/graph/state.py`:

| Key | Written by | Read by |
| --- | --- | --- |
| `case` | (input) | all |
| `analysis` | case_analysis | policy_evidence, coverage, decision |
| `queries` | case_analysis | policy_evidence |
| `evidence` | policy_evidence | coverage_exclusion, validation |
| `retrieval_stats` | policy_evidence | decision, trace |
| `assessment` | coverage_exclusion | decision, validation |
| `decision_draft` | decision | validation |
| `validation` | validation | decision (on revision) |
| `trace` | all (append-only reducer) | API response |

The conditional edge after validation implements the retry behaviour: `FAIL`
routes back to Decision, which re-scores with the validation result folded into
confidence. A second `FAIL` terminates in `NEEDS_REVIEW`.

If `langgraph` is unavailable, an equivalent built-in state machine runs the
same graph with identical semantics, so the system stays deployable on a minimal
image.

---

## 6. RAG pipeline

```
Policy PDF (17 pages)
   → ingestion        page-numbered text, boilerplate stripped empirically
   → chunking         structure-aware → 137 chunks with full metadata
   → indexing         dense embeddings + BM25, persisted with a content hash
   → query            one topical query per dimension
                      + one anchor query per clause a rule needs
   → dense + BM25     parallel arms
   → RRF fusion       rank-based, scale-invariant
   → reranking        query-document interaction scoring
   → selection        per-dimension recall floor, then precision fill
   → evidence binding rules resolve their anchors or abstain
```

---

## 7. Chunking strategy

**Not fixed-size windows.** The policy is a legal instrument whose meaning lives
in atomic units — a definition, a numbered exclusion, a sub-limit note. A 512-
character window slices those in half, and a retriever then returns the bottom of
exclusion 4 and the top of exclusion 5, leaving the reasoning layer to guess
which clause it has.

The chunker segments by structure and applies size control only *within* a unit:

1. Lines tagged with their source page — provenance survives chunking.
2. Running headers/footers detected **empirically** (a line on ≥60% of pages is
   boilerplate) rather than by line index.
3. Top-level sections detected from the wording's own headings.
4. Each section segmented by the unit type it actually uses:
   - **Definitions** → one chunk per defined term, headed by the term
   - **What We Exclude** → one chunk per numbered exclusion
   - **Scope of Cover** → numbered benefits, `NB1–NB5` notes, `Sub limits`, `Note`
   - **Standard Terms** → one chunk per numbered condition
5. Oversized units split on sentence boundaries with overlap. Units are **never
   merged across a clause boundary** — merging two exclusions makes a citation
   ambiguous.

**Result:**

| Section | Chunks |
| --- | --- |
| Definitions | 58 |
| Standard Terms and Conditions | 26 |
| What We Exclude | 21 *(exactly matching the policy's 21 numbered exclusions)* |
| Scope of Cover | 14 |
| Critical Illness Definitions | 6 |
| Claims Procedure | 6 |
| Other | 6 |
| **Total** | **137** |

Length spread 42–1564 characters (median 246) — the non-uniformity confirms the
chunker follows structure, not a size budget. `tests/test_chunking.py` asserts
that 14 decisive clauses survive chunking intact **and land on their correct
page**.

Each chunk carries `chunk_id`, `page`, `section`, `heading`, `clause_ref`,
`source`, `char_count`, `token_estimate`. For indexing only, the section and
heading are prefixed to the text (contextual retrieval); the stored `text` stays
pristine so citations quote the policy verbatim.

---

## 8. Dense retrieval

Pluggable, selected by `EMBEDDING_BACKEND`:

| Backend | When | Notes |
| --- | --- | --- |
| `sentence_transformer` | Weights reachable | BGE-small by default |
| `lsa` | Offline default | TF-IDF (word 1–2-gram + char 3–5-gram) → truncated SVD |

Both emit L2-normalised dense vectors scored by cosine similarity, so the rest
of the pipeline is unchanged. `auto` probes for the transformer and falls back
silently.

**Why a fallback exists at all.** This was not a preference. In the build
environment the model hub returned `403` on every request, and free-tier hosts
are frequently locked down the same way — a 400 MB download is also a poor fit
for a 512 MB dyno. Rather than let the system be undeployable, the dense arm is
an interface with two implementations.

LSA is genuinely dense and genuinely semantic: SVD places *domiciliary* near
*confined at home* through co-occurrence. It is corpus-fitted rather than
pre-trained, so it generalises less well to paraphrases the policy never uses —
which is precisely what the lexical arm and anchor-directed retrieval
compensate for. Character n-grams also absorb the British/American spelling
split that runs through this document (*hospitalisation* / *hospitalization*).

**Vector index.** At 137 chunks an exact cosine scan over a contiguous `float32`
matrix is a single BLAS call — faster than any ANN structure once build time is
counted, and exact by construction. FAISS is used automatically if installed.

---

## 9. BM25 retrieval

Okapi BM25 (`k1=1.5`, `b=0.75`) implemented in-tree rather than pulled from
`rank_bm25`, for two domain reasons:

1. **The tokenizer must keep numbers and units.** `1.0%`, `48`, `24 hours`,
   `pre-existing` *are* the decisive evidence in this policy. A generic
   tokenizer discards or mangles them.
2. **Per-term score attribution** is needed to explain why a chunk matched, and
   to let the reranker weight rare terms — the library does not expose it.

IDF uses the `+1` guarded form, which keeps common terms from going negative on
a small corpus (137 chunks).

---

## 10. Fusion

**Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, SIGIR 2009):

```
score(d) = Σ_arms  weight_arm / (RRF_K + rank_arm(d))
```

**Why RRF rather than a weighted score sum.** BM25 scores are unbounded and
corpus-dependent; cosine sits in [-1, 1]. Normalising them onto a shared scale
needs calibration that would have to be re-tuned whenever the embedding backend
changes — and this system has a *pluggable* backend. RRF consumes only ranks, so
it is invariant to both. A small normalised-score term breaks ties within a rank.
`tests/test_retrieval.py::test_rrf_is_scale_invariant` asserts that multiplying
BM25 scores by 1000 does not change the fused order.

Fusion runs twice: once per query across the two arms, then across all queries
(`merge_multi_query`) so a chunk relevant to several dimensions accumulates
reciprocal-rank mass from each.

---

## 11. Reranking

A reranker's defining property is that it scores query and document **together**
— unlike a bi-encoder's independent embedding or BM25's bag-of-words assumption.
Whether that function is a neural network is an implementation choice.

| Backend | When |
| --- | --- |
| `cross_encoder` | BGE cross-encoder, when weights load |
| `lexical` | Offline default |

`LexicalSemanticReranker` scores interaction through:

| Signal | Weight | Why |
| --- | --- | --- |
| IDF-weighted term coverage | 0.42 | Rare policy terms carry the meaning |
| Bigram/phrase adjacency | 0.18 | "waiting period" ≠ "waiting" + "period" |
| Numeric-and-unit agreement | 0.16 | "48 months" matching "48 months" is near-decisive |
| Section prior by dimension | 0.14 | Encodes *this document's* layout, not insurance knowledge |
| Dense similarity | 0.10 | One input among several |

Scores are squashed through a logistic so `MIN_RERANK_SCORE` means the same
thing regardless of which dense backend produced the candidate.

**One domain correction** (failure F-05): the short-text penalty does **not**
apply to chunks carrying a `clause_ref`. `7. Dental treatment or surgery of any
kind.` is 43 characters and disposes of a claim completely; penalising it as a
fragment pushed it out of retrieval entirely.

---

## 12. Citation design

```json
{
  "claim": "Claim falls within the 30-day initial waiting period: only 19 day(s) elapsed…",
  "source": "USGIC-CSC-Individual-Health-Insurance.pdf",
  "page": 9,
  "section": "What We Exclude",
  "heading": "Exclusion 2: 30 days Waiting Period",
  "chunk_id": "p09-exclusion-2-30-days-waiting-period-080",
  "quote": "…A waiting period of 30 days will apply to all claims unless…",
  "rule_id": "waiting_period_initial_30d"
}
```

Citations are **constructed from retrieved chunks, never assembled**:

- `page`, `section`, `heading`, `chunk_id` are copied from the resolved chunk.
- `quote` is *sliced* out of the chunk text around the anchor — there is no code
  path that composes quote text.
- `rule_id` links back to the rule that relied on it.

Verified three ways: at runtime by `citation_integrity` and `quote_provenance`;
in the metrics by `citation_correctness`; and in
`tests/test_validation.py` by deliberately fabricating a citation to page 99 and
asserting it is caught.

---

## 13. Abstention strategy

Abstention triggers, in order of precedence:

1. **A condition precedent cannot be established** — facility not evidenced as a
   Hospital, medical necessity unconfirmed, no domiciliary qualifying
   circumstance, pre-existing status unknown.
2. **An anchor clause a relevant rule needs was not retrieved.**
3. **Confidence below `ABSTAIN_BELOW_CONFIDENCE`** on an affirmative decision.
4. **Validation failure** surviving the revision budget.

**`null` vs absent.** An explicit `null` in `evidence_context` means *the
question was raised and could not be established* → abstain. An **absent** field
means *not in issue* → do not raise it. This is what separates PUB-006 (which
carries `"medical_necessity_confirmed": null` and abstains) from PUB-001 (which
does not mention it and does not).

**Blocking outranks abstention.** If cosmetic surgery is excluded, an unresolved
question about facility registration does not change the outcome. Abstaining
there would be noise, not caution.

**Only affirmative decisions are downgraded on low confidence.** A rejection
already grounded in a cited blocking clause is not made safer by conversion to a
review.

Measured abstention performance: **precision 100%, recall 100%, F1 100%** —
4 expected abstentions, 4 produced, 0 missed, 0 spurious.

---

## 14. Confidence calculation

Confidence measures **evidential support**, not model certainty.

```
raw   = 0.25 · retrieval_quality
      + 0.25 · evidence_coverage
      + 0.25 · citation_support
      + 0.25 · decision_consistency

final = clamp(raw × validation_factor − missing_evidence_penalty, 0, 1)
```

| Component | Definition |
| --- | --- |
| `retrieval_quality` | Mean rerank score of the evidence findings actually cited (falls back to top-k mean) |
| `evidence_coverage` | Share of investigated dimensions that returned ≥1 chunk |
| `citation_support` | Share of material findings carrying ≥1 citation |
| `decision_consistency` | Whether findings agree with the decision; contradictory signals reduce it |
| `validation_factor` | `1.0` on PASS, `0.55` on FAIL |
| `missing_evidence_penalty` | `0.08` per blocking gap, `0.03` per non-blocking, capped at `0.35` |

`citation_support` sits near 1.0 by construction — unsupported findings cannot
be produced. It is kept as a **regression alarm**: if it drops, evidence binding
has broken somewhere.

The full breakdown is returned in every response (`confidence_breakdown`) and
rendered in the UI, so a reviewer can see exactly which term lowered a score.

Observed calibration: mean confidence **0.96** on correct decisions; the four
abstention cases score **0.68–0.89**, i.e. the system is measurably less
confident exactly where it declines to decide.

---

## 15. API documentation

Interactive docs at `/docs`. Base URL local: `http://localhost:8000`.

### `GET /health`

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "index_ready": true,
  "policy_indexed": true,
  "chunks_indexed": 137,
  "policy_source": "USGIC-CSC-Individual-Health-Insurance.pdf",
  "policy_id": "UNIHLIP18004V011718",
  "backends": {
    "embedding": "lsa", "dense_index": "numpy", "sparse": "bm25",
    "reranker": "lexical_semantic", "chunks": "137",
    "orchestrator": "langgraph", "llm": "disabled"
  },
  "uptime_seconds": 12.4
}
```

Returns **200 with `status: "degraded"`** (not an error) when the index is
missing, so a platform health check can distinguish "up but not ready" from
"broken".

### `POST /analyze`

Accepts a claim either bare (the supplied dataset's own shape) or wrapped in
`{"case": {...}}`, so a reviewer can paste a case straight from
`public_test_cases.json`.

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "case_id": "PUB-001",
    "policy_start_date": "2025-01-01",
    "claim_date": "2026-03-14",
    "sum_insured_inr": 500000,
    "continuous_coverage_months": 14,
    "patient": {"age": 34},
    "hospital": {"name": "Sunrise Multispeciality", "network_provider": true},
    "treatment": {"type": "inpatient", "admission_hours": 96,
                  "diagnosis": "Acute appendicitis", "procedure": "Appendectomy",
                  "pre_existing": false, "experimental": false},
    "expenses_inr": {"room": 30000, "doctor_fees": 30000,
                     "medicines_diagnostics": 90000, "pre_hospitalization": 5000,
                     "post_hospitalization": 7000, "ambulance": 1200},
    "documents": ["claim_form", "discharge_summary", "itemized_bill"]
  }'
```

Response (abridged):

```json
{
  "case_id": "PUB-001",
  "decision": "ADMISSIBLE_WITH_LIMITS",
  "confidence": 0.97,
  "summary": "Claim PUB-001 is admissible subject to policy limits on room, ambulance. Total deduction INR 10,200 against a claimed INR 163,200. 1 evidence item(s) noted but not decisive.",
  "key_findings": [
    {
      "rule_id": "limit_room_rent",
      "dimension": "room_rent_limit",
      "status": "LIMIT_APPLIES",
      "severity": "LIMITING",
      "statement": "Normal room expenses capped at 1.0% of Basic Sum Insured per day…",
      "evidence_chunk_ids": ["p07-sub-limits-067"],
      "citations": [{ "page": 7, "section": "Scope of Cover", "chunk_id": "p07-sub-limits-067" }]
    }
  ],
  "applicable_limits": [
    {
      "limit_id": "limit_room_rent",
      "category": "room",
      "basis": "1.00% of Sum Insured (INR 500,000) = INR 5,000/day x 4 day(s) = INR 20,000",
      "limit_amount_inr": 20000, "claimed_amount_inr": 30000,
      "allowed_amount_inr": 20000, "deduction_inr": 10000, "binding": true
    }
  ],
  "missing_evidence": [ "…1 non-blocking item…" ],
  "citations": [ "…12 citations…" ],
  "validation": { "status": "PASS", "unsupported_claims": [], "revision_required": false },
  "confidence_breakdown": {
    "retrieval_quality": 0.986, "evidence_coverage": 1.0,
    "citation_support": 1.0, "decision_consistency": 1.0,
    "raw_score": 0.996, "final_score": 0.966
  },
  "payable_estimate_inr": 153000,
  "claimed_total_inr": 163200,
  "total_deduction_inr": 10200,
  "abstained": false,
  "trace": [
    {"agent": "case_analysis", "action": "Extracted claim facts and built investigation plan",
     "elapsed_ms": 3, "metrics": {"decision_dimensions": 12, "investigation_questions": 19}},
    {"agent": "policy_evidence", "action": "Hybrid retrieval with fusion and reranking",
     "elapsed_ms": 148, "metrics": {"dense_results": 227, "bm25_results": 225,
                                    "fused_results": 32, "reranked_results": 28}},
    {"agent": "coverage_exclusion", "action": "Evaluated coverage, waiting periods, exclusions and limits",
     "elapsed_ms": 4, "metrics": {"findings": 9, "binding_limits": 2}},
    {"agent": "decision", "action": "Generated structured decision", "elapsed_ms": 1},
    {"agent": "validation", "action": "Verified statements against evidence",
     "status": "PASS", "elapsed_ms": 1, "metrics": {"checks_run": 7}}
  ]
}
```

Query flags: `include_evidence` (default `true`), `include_trace` (default `true`).

### `POST /analyze/batch`

```bash
curl -s -X POST http://localhost:8000/analyze/batch \
  -H 'Content-Type: application/json' \
  -d '{"cases": [ ... up to 50 ... ], "include_evidence": false}'
```

### `GET /index/info`

Lets a reviewer verify chunking quality without reading the code — chunk counts
by section, pages indexed, active backends, and a chunk sample.

### Error handling

| Situation | Status | Body |
| --- | --- | --- |
| Malformed JSON | 400/422 | `error`, `detail`, `hint` |
| Schema violation | 422 | `field_errors[]` naming each bad field |
| Unknown/irrelevant fields | **200** | Tolerated and ignored (RULE 9) |
| Body over `MAX_REQUEST_BYTES` | 413 | Size limit stated |
| Index unavailable | 503 | Names the ingestion command to run |
| Internal error | 500 | Error type only — never a traceback |
| Pipeline failure mid-analysis | **200** | Valid `NEEDS_REVIEW` with the error in the trace |

That last row is deliberate: a claims system that crashes tells the reviewer
nothing, whereas an abstention with the failure recorded is actionable.

---

## 16. Frontend

Streamlit, built for a reviewer who must defend the decision.

- **Sidebar** — select any public or custom case, paste JSON, or upload a file.
- **Decision banner** — colour-coded status, summary, confidence, validation,
  elapsed time. `NEEDS_REVIEW` gets a dedicated panel stating the abstention
  reason, because "I cannot safely decide" is a first-class answer here.
- **Metrics row** — confidence, claimed, deductions, payable estimate, citations.
- **Findings** — decisive findings first, each expandable to its citations with
  page, section, `chunk_id` and the verbatim quote.
- **Limits & deductions** — table plus the arithmetic basis for each cap.
- **Missing evidence** — blocking vs non-blocking, with the document that would
  resolve each.
- **Policy evidence** — every retrieved chunk with dense/BM25/fusion/rerank
  scores, retrieval method and dimension attribution; filterable to cited chunks.
- **Execution trace** — agent, action, status, timings, retrieval counts, plus
  the confidence breakdown showing which term lowered the score.
- **Validation** — checks run, checks failed, unsupported claims.
- **Investigation** — the plan, missing input fields, and the attributes the
  system considered and discounted as policy-irrelevant.
- **Raw JSON** — full response, downloadable.

Runs against a deployed backend when `API_BASE_URL` is set, otherwise in-process
— so it works standalone on Streamlit Community Cloud.

---

## 17. Local setup

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd aptino-claim-engine

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # optional; every value has a default

python -m scripts.ingest_policy    # build the index (~6s, one time)
uvicorn app.api.main:app --reload  # terminal 1
streamlit run frontend/streamlit_app.py   # terminal 2
```

API at `http://localhost:8000` (docs at `/docs`), UI at `http://localhost:8501`.

Optional transformer upgrade, on a machine that can reach the model hub:

```bash
pip install sentence-transformers faiss-cpu
EMBEDDING_BACKEND=sentence_transformer RERANKER_BACKEND=cross_encoder \
  python -m scripts.ingest_policy --force
```

`make help` lists shortcuts for all of the above.

---

## 18. Environment variables

All optional — the system runs with no `.env` at all. Full annotated list in
[`.env.example`](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `POLICY_PDF_PATH` | `data/policy/USGIC-…pdf` | Source policy |
| `INDEX_DIR` | `data/index` | Where the index is persisted |
| `PORT` | `8000` | API port (PaaS providers inject this) |
| `CORS_ORIGINS` | `*` | Comma-separated origins |
| `API_BASE_URL` | *(empty)* | Frontend → backend; empty means in-process |
| `EMBEDDING_BACKEND` | `auto` | `auto` / `sentence_transformer` / `lsa` |
| `RERANKER_BACKEND` | `auto` | `auto` / `cross_encoder` / `lexical` |
| `PER_DIMENSION_K` | `3` | Evidence floor per dimension |
| `RRF_K` | `60` | Fusion smoothing constant |
| `MIN_RERANK_SCORE` | `0.12` | Usable-evidence threshold |
| `ABSTAIN_BELOW_CONFIDENCE` | `0.45` | Downgrade threshold |
| `MAX_VALIDATION_REVISIONS` | `1` | Revision budget |
| `LLM_PROVIDER` | `none` | `none` / `openai` / `openrouter` / `groq` / `together` / `ollama` |
| `LLM_API_KEY` | *(empty)* | **Never committed** — environment only |

---

## 19. Running ingestion

```bash
python -m scripts.ingest_policy                  # build or load
python -m scripts.ingest_policy --force          # rebuild
python -m scripts.ingest_policy --inspect        # chunk breakdown
python -m scripts.ingest_policy --query "room rent limit per day"
```

The index stores a SHA-256 fingerprint of the policy PDF and rebuilds
automatically if the document changes.

---

## 20. Running the API

```bash
uvicorn app.api.main:app --reload                       # development
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 2   # production
make api
```

---

## 21. Running the frontend

```bash
streamlit run frontend/streamlit_app.py                       # in-process engine
API_BASE_URL=https://your-api.onrender.com \
  streamlit run frontend/streamlit_app.py                     # against deployed API
make ui
```

---

## 22. Running the evaluation

```bash
python -m evaluation.run_evaluation                  # everything
python -m evaluation.run_evaluation --suite public   # 12 supplied cases
python -m evaluation.run_evaluation --suite custom   # 10 candidate cases
python -m evaluation.run_evaluation --case PUB-011   # one case
make eval
```

Writes `results.json`, `results.csv`, `summary.md` and `decisions.json` to
`evaluation/results/`.

---

## 23. Test commands

```bash
python -m pytest tests/ -v                    # all 152 tests
python -m pytest tests/test_chunking.py -v    # chunk metadata, structure, clause integrity
python -m pytest tests/test_retrieval.py -v   # BM25, dense, fusion, reranking, hybrid
python -m pytest tests/test_agents.py -v      # agent behaviour, evidence binding, state
python -m pytest tests/test_validation.py -v  # tampering detection, abstention
python -m pytest tests/test_api.py -v         # contract, error handling, determinism
python -m pytest tests/test_evaluation.py -v  # metrics + end-to-end accuracy gate
make test
```

`tests/test_evaluation.py::TestEndToEndAccuracy` is the regression gate: a
retrieval or rule change that breaks a case fails here rather than silently
degrading the reported metrics.

---

## 24. Deployment

### Docker (both services)

```bash
docker compose up --build
# API http://localhost:8000 · UI http://localhost:8501
```

The image builds the index at **build time**, so containers start ready. It runs
as a non-root user and ships a `HEALTHCHECK`.

### Render (backend) — free tier

`render.yaml` is a ready blueprint: **New → Blueprint → point at this repo.**

It pins `EMBEDDING_BACKEND=lsa` and `RERANKER_BACKEND=lexical` because the free
tier's 512 MB cannot hold transformer weights alongside the service. Add
`LLM_API_KEY` as a dashboard secret only if you want the optional LLM layer —
never in the file.

Set `CORS_ORIGINS` to your Streamlit origin once the frontend is deployed.

### Streamlit Community Cloud (frontend)

Point it at `frontend/streamlit_app.py`. In **Advanced settings → Secrets**:

```toml
API_BASE_URL = "https://your-api.onrender.com"
```

Omit it to run the engine in-process — the app works standalone either way.

> **Deliverable note.** The live URLs are not filled in here because deployment
> requires account credentials I do not have. Everything needed is committed
> (`Dockerfile`, `docker-compose.yml`, `render.yaml`, `.streamlit/config.toml`),
> both services have been verified running locally, and the blueprint deploys as
> a one-click action. See [§30](#30-remaining-manual-steps).

---

## 25. Evaluation results

**22 cases: the 12 supplied public cases (unmodified) + 10 candidate-authored.**

### Headline

| Metric | Result |
| --- | --- |
| Decision accuracy | **100.0%** (22/22) |
| Deduction accuracy | **100.0%** (11 cases with a monetary label) |
| Evidence recall@k | **100.0%** |
| Citation hit rate | **100.0%** |
| Citation correctness | **100.0%** |
| Validation pass rate | **100.0%** |
| Abstention P / R / F1 | **100% / 100% / 100%** |
| Mean latency | **264 ms** (median 249 ms, min 165 ms, max 572 ms) |

Latency figures match the checked-in `evaluation/results/summary.md` and move
by roughly ±10% between runs on the same machine (two audit runs gave 243 ms
and 264 ms); the max is dominated by first-call warm-up. They are
hardware-dependent and were measured with `LLM_PROVIDER=none` on
the offline backends (`embedding=lsa`, `reranker=lexical_semantic`), which is
the reproducible configuration. Enabling an LLM does not change any decision,
deduction or citation — it only rewrites the summary prose — but it dominates
wall-clock time: a local `ollama:llama3.1` run takes 60–105 s per claim.

### How expected outcomes were established

Each label was derived **by hand from the policy wording before the engine was
run**, by locating the governing clause and applying it to the case facts.
`evaluation/expected_results/*.json` records the exact provision and page for
every case, so a reviewer can verify any label independently. Example:

> **PUB-003 → NOT_ADMISSIBLE.** Exclusion 1, p.8: pre-existing diseases are not
> covered until 48 months of continuous coverage have elapsed. The case declares
> `pre_existing: true` with 27 months and no prior-insurer credit — 21 months
> short.

**The engine has no access to these files.** It derives decisions from retrieved
policy text via the rule engine; the harness compares afterwards.

### Public suite (12 supplied cases)

| Case | Expected | Actual | ✓ | Conf | Tests |
| --- | --- | --- | --- | --- | --- |
| PUB-001 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 0.97 | Room + ambulance caps |
| PUB-002 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 30-day waiting period |
| PUB-003 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 0.97 | 48-month PED waiting |
| PUB-004 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 1.00 | Domiciliary 20% sub-limit |
| PUB-005 | ADMISSIBLE | ADMISSIBLE | ✅ | 0.97 | Day care < 24 hours |
| PUB-006 | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.86 | **Insufficient evidence** |
| PUB-007 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 0.97 | Four caps, INR 190,500 |
| PUB-008 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | Cosmetic exclusion |
| PUB-009 | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | Pre/post windows (boundary) |
| PUB-010 | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | Portability waiver (multi-clause) |
| PUB-011 | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.89 | **Hospital definition gap** |
| PUB-012 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | Experimental treatment |

### Custom suite (10 candidate-authored cases)

Written to reach behaviours the supplied 12 do not: a per-day ICU cap, several
caps binding at once, `PARTIALLY_ADMISSIBLE`, continuity credit that is granted
but still insufficient, two clauses pulling in opposite directions, and
robustness to irrelevant attributes.

| Case | Expected | Actual | ✓ | Scenario |
| --- | --- | --- | --- | --- |
| CUS-001 | ADMISSIBLE | ADMISSIBLE | ✅ | Clean claim, no cap binds |
| CUS-002 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 4 caps at once (ICU per-day), INR 89,500 |
| CUS-003 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | Dental exclusion (43-char clause) |
| CUS-004 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | Hernia in first policy year |
| CUS-005 | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | **Three conditions unresolved** |
| CUS-006 | PARTIALLY_ADMISSIBLE | PARTIALLY_ADMISSIBLE | ✅ | Pre/post outside 30/60-day windows |
| CUS-007 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 3 credited years, still 42 < 48 months |
| CUS-008 | ADMISSIBLE | ADMISSIBLE | ✅ | Dialysis: covered *and* on the waiting list |
| CUS-009 | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | **Domiciliary circumstance unevidenced** |
| CUS-010 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 8 irrelevant attributes ignored |

All five decision statuses are exercised; six cases require abstention or
multi-clause reasoning.

### Confidence calibration

Mean confidence **0.96** on correct decisions; the four abstention cases score
**0.68–0.89** — the system is measurably least confident exactly where it
declines to decide.

---

## 26. Failure analysis

Seven real defects found and fixed during development, each with root cause and
regression test. Full detail: [`docs/failure-analysis.md`](docs/failure-analysis.md).

**Decision accuracy across the fixes: 77.3% → 95.5% → 100.0%.**

| ID | Failure | Root cause | Result |
| --- | --- | --- | --- |
| **F-01** | Rules abstained because their clause was never retrieved | Global evidence pool starved narrow dimensions | Two-phase selection with a per-dimension recall floor |
| **F-02** | Exclusion 20's scope is ambiguous in the source PDF | OCR flattened a nested list | Abstain on the ambiguous branch; decide where both readings agree |
| **F-03** | Domiciliary "three day" rule unevaluable | Input schema has no treatment duration | Documented limitation + non-blocking gap |
| **F-04** | **Five cases over-abstained after a "fix"** | One dimension serving two information needs | Split into its own dimension → 77.3% → 95.5% |
| **F-05** | A 43-char decisive exclusion was ranked out | Short-text penalty applied to complete clauses | Exempt chunks carrying a `clause_ref` |
| **F-06** | Short precise clauses lost to broad queries | No query targeted the clause a rule needed | **Anchor-directed retrieval** → 95.5% → 100% |
| **F-07** | Infinite loop in validation → decision | Revision counter not incremented on terminal path | Both loop guards now agree |

**The most instructive was F-04.** Adding documentation terms to the
`SCOPE_OF_COVER` query diluted it until the scope clause dropped out of its own
dimension's results. Five cases flipped to `NEEDS_REVIEW` — **none produced a
wrong answer.** The architecture converted a retrieval regression into
abstention, which is the correct failure mode for claims adjudication and made
the bug obvious in the evaluation output.

**F-06 produced the most valuable idea:** since every rule already declares the
clause it depends on, those declarations can *drive* retrieval. Broad topical
queries find the right area of the policy; anchor queries find the exact clause.
That took `evidence_recall@k` from 93.2% to 100%.

---

## 27. Design trade-offs

| Decision | Chosen | Alternative | Why |
| --- | --- | --- | --- |
| Decision logic | Deterministic rules bound to evidence | LLM reasoning | Exact arithmetic, reproducible evaluation, auditable clause links |
| LLM role | Optional, non-decisional | Central | Identical decisions with no key; no non-determinism in adjudication |
| Dense backend | Pluggable, LSA default | BGE only | Model hub returned 403 in the build environment; free tiers are similarly locked down |
| Reranker | Pluggable, lexical default | Cross-encoder only | Deterministic and reproducible in CI; transformer used when available |
| BM25 | In-tree | `rank_bm25` | Tokenizer must keep "1.0%", "48", "24 hours"; per-term attribution needed |
| Vector index | numpy exact, FAISS optional | FAISS always | 137 chunks: exact search is one BLAS call and exact by construction |
| Fusion | RRF | Weighted score sum | Scale-invariant, survives a backend swap without recalibration |
| Chunking | Structure-aware | Fixed-size windows | Citations must point at whole clauses |
| Ambiguous clause | Abstain on the ambiguous branch | Pick a reading | The document does not settle it |
| Error handling | Failures become `NEEDS_REVIEW` | Propagate 500 | A crash tells a reviewer nothing; an abstention is actionable |

---

## 28. Known limitations

Stated plainly, because the headline numbers should not be read without them.

1. **22 cases is a small evaluation set.** 100% accuracy here is not evidence of
   100% accuracy in general. The confidence interval on 22 samples is wide.
2. **I wrote both the rules and the custom cases.** That is a genuine source of
   optimism: my cases test the behaviours I thought of. An independent case
   author would likely find gaps. The 12 supplied cases are the more meaningful
   signal, and they were not written by me.
3. **The ground truth is my reading of the policy**, not an adjuster's. It is
   documented clause by clause so it can be challenged, and F-03 records the
   judgement I am least sure of.
4. **LSA embeddings generalise poorly to unseen paraphrase.** Fitted on 137
   chunks, they represent *this* policy's vocabulary well. A claim described in
   wholly different language would lean on the BM25 arm and anchor-directed
   retrieval. `EMBEDDING_BACKEND=sentence_transformer` addresses this where the
   hub is reachable.
5. **The policy PDF has a structural defect** (F-02): exclusions 17–20 are
   rendered flat when the wording indicates nesting. The system abstains on the
   ambiguous branch rather than guessing, but the ambiguity is unresolvable from
   the document.
6. **The rule set covers the provisions these 22 cases engage**, not the entire
   policy. Critical Illness cover, cumulative bonus, portability mechanics,
   contribution and multiple-policy clauses are indexed and retrievable but have
   no dedicated rules — a claim turning on them would abstain rather than answer.
7. **Payable amounts are estimates, not authorisations.** They show the financial
   effect of the limits identified; they do not model co-pay, cumulative bonus,
   or prior claims against the sum insured.
8. **No authentication or rate limiting.** Appropriate for a review deployment,
   not for production (see §29).
9. **Single-policy system.** `policy_id` is carried through but one index is
   built for one wording; multi-policy support would need per-policy indices.
10. **The LLM path is implemented but lightly exercised**, since no key was
    available in the build environment. Its failure modes are handled (every call
    degrades to the deterministic path) but its *quality* contribution is
    untested.

---

## 29. Security considerations

- **No secrets in the repository.** All credentials come from environment
  variables; `.env` is git-ignored and `.env.example` contains no real values.
  `render.yaml` marks `LLM_API_KEY` as `sync: false` so it must be set in the
  dashboard.
- **Input validation** on every field via Pydantic, with explicit bounds
  (`sum_insured_inr > 0`, `0 ≤ age ≤ 120`, `admission_hours ≥ 0`) and a request
  size cap.
- **No traceback leakage.** The 500 handler returns the exception *type* only;
  details go to server logs.
- **Unknown fields are tolerated but never executed** — extra attributes are
  carried for transparency and explicitly excluded from decision logic.
- **Container runs as non-root** (uid 10001).
- **CORS is configurable**; set `CORS_ORIGINS` to your frontend origin rather
  than leaving `*` in production.
- **No PII by design.** The dataset is synthetic, and the system stores nothing:
  each request is stateless and no claim data is persisted.
- **Not included, and needed before production:** authentication, rate limiting,
  audit logging of decisions, and encryption at rest for any stored claim data.

---

## 30. Remaining manual steps

Everything below needs account credentials rather than code:

1. **Initialise git and push to a public GitHub repository.** The working
   tree is complete, but it is not yet a git repository — `git init` has not
   been run, so nothing is under version control. Both Render and Streamlit
   Community Cloud deploy *from a repository*, so this is a prerequisite for
   steps 2 and 3. `.gitignore` already excludes `.env`; verify with
   `git status --porcelain --ignored | grep '\.env'` before the first push.
2. **Deploy the backend** — Render: New → Blueprint → select this repo.
   `render.yaml` handles the rest.
3. **Deploy the frontend** — Streamlit Community Cloud, pointed at
   `frontend/streamlit_app.py`, with `API_BASE_URL` set to the Render URL.
4. **Tighten CORS** — set `CORS_ORIGINS` to the Streamlit origin.
5. *(Optional)* Add `LLM_API_KEY` to enable summary polishing and the advisory
   second-opinion validation pass.

Both services have been verified running locally (API on `:8000` returning
`status: ok` with 137 chunks indexed; Streamlit on `:8501` returning HTTP 200).

---

## Project structure

```
aptino-claim-engine/
├── app/
│   ├── config.py                  env-driven settings, no hard-coded secrets
│   ├── api/
│   │   ├── main.py                FastAPI app, CORS, error handlers, lifespan
│   │   ├── routes.py              /analyze · /analyze/batch · /health · /index/info
│   │   └── schemas.py             request/response envelopes
│   ├── agents/
│   │   ├── base.py                trace timing (no chain-of-thought by construction)
│   │   ├── case_analysis.py       AGENT 1 — facts, dimensions, queries
│   │   ├── policy_evidence.py     AGENT 2 — retrieval stack
│   │   ├── coverage_exclusion.py  AGENT 3 — rules + limits
│   │   ├── decision.py            AGENT 4 — precedence + confidence
│   │   └── validation.py          AGENT 5 — 7 verification checks
│   ├── graph/
│   │   ├── state.py               typed state + ownership table
│   │   └── workflow.py            LangGraph wiring + revision loop + fallback runner
│   ├── retrieval/
│   │   ├── ingestion.py           PDF → page-numbered text
│   │   ├── chunking.py            structure-aware chunker
│   │   ├── embeddings.py          pluggable dense backends
│   │   ├── dense.py               vector index (numpy / FAISS)
│   │   ├── bm25.py                Okapi BM25, policy-tuned tokenizer
│   │   ├── fusion.py              RRF + multi-query merge
│   │   ├── reranker.py            pluggable rerankers
│   │   └── index.py               the hybrid retriever
│   ├── policy/
│   │   ├── evidence_binding.py    anchors, resolution, citation construction
│   │   ├── rules.py               the rule engine
│   │   └── limits.py              monetary cap calculators
│   ├── models/schemas.py          the inter-agent contract
│   ├── services/
│   │   ├── engine.py              facade used by API, evaluation and tests
│   │   ├── confidence.py          documented confidence model
│   │   └── llm.py                 optional LLM client
│   └── utils/                     text normalisation, date arithmetic
├── frontend/streamlit_app.py
├── evaluation/
│   ├── public_cases/              12 supplied cases, byte-identical
│   ├── custom_cases/              10 candidate-authored cases
│   ├── expected_results/          clause-justified ground truth
│   ├── metrics.py
│   ├── run_evaluation.py
│   └── results/                   results.json · results.csv · summary.md
├── tests/                         152 tests across 6 modules
├── docs/
│   ├── architecture.md            design note
│   ├── failure-analysis.md        7 failures, root causes, fixes
│   └── claim_case_schema.md       supplied input schema
├── scripts/ingest_policy.py
├── data/policy/                   the supplied policy PDF
├── Dockerfile · docker-compose.yml · render.yaml · Makefile
├── requirements.txt · pyproject.toml · .env.example · .gitignore
└── README.md
```

---

## Acceptance checklist

| Requirement | Status |
| --- | --- |
| Policy indexed with meaningful chunks and page/section metadata | ✅ 137 structure-aware chunks |
| Dense + sparse retrieval + reranking | ✅ LSA/BGE + BM25 + RRF + reranker |
| At least three genuinely specialised agents | ✅ Five, separately tested |
| Agents exchange structured state | ✅ Typed Pydantic models |
| Final decision structured and machine-readable | ✅ `AnalysisResponse` |
| Material claims have inspectable citations | ✅ Page, section, chunk id, quote |
| System abstains when evidence is insufficient | ✅ Abstention F1 100% |
| API and frontend usable | ✅ Both verified running |
| Execution trace without chain-of-thought | ✅ Metrics only, asserted by test |
| All supplied public cases evaluated | ✅ 12/12, unmodified |
| At least five additional cases evaluated | ✅ 10 |
| At least two NEEDS_REVIEW cases | ✅ 4 |
| At least three failure cases documented | ✅ 7 |
| Deployed and locally reproducible | ⚠️ Reproducible locally + deploy config ready; live URLs need credentials (§30) |

---

*Built against Universal Sompo CSC Individual Health Insurance, policy wording
UNIHLIP18004V011718. All claim data is synthetic.*
