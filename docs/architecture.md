# Architecture & Design Note

Policy-Aware Multi-Agent RAG Claim Decision Engine — design rationale, agent
boundaries, state flow, retrieval architecture and the trade-offs behind them.

---

## 1. The central design decision

An insurance claim decision is not a generative task. Whether 27 months of
coverage clears a 48-month waiting period is arithmetic; whether cosmetic
surgery is excluded is a lookup. What is genuinely hard is **finding the right
clause in a 17-page legal document, combining clauses that interact, and knowing
when the document does not answer the question.**

So the system splits the problem along that line:

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Retrieval | Find the governing clauses | Hybrid dense + BM25 → RRF → rerank |
| Reasoning | Apply them to the facts | Deterministic rule engine, evidence-bound |
| Verification | Confirm the output is supported | Independent validation agent |
| Language | Phrase the result | Optional LLM, cannot change a decision |

The LLM is deliberately kept out of the decision path. It polishes the summary,
proposes extra retrieval phrasings, and offers an advisory second opinion during
validation. **Decisions are byte-identical with `LLM_PROVIDER=none`**, which is
what makes the evaluation reproducible and lets the system run at zero marginal
cost.

This is not an avoidance of LLM usage; it is a claim about where non-determinism
belongs. An adjuster must be able to defend a decision clause by clause, and a
sampled generation cannot offer that guarantee.

---

## 2. Evidence binding — the load-bearing mechanism

Every policy rule declares the **anchor text** it depends on: a verbatim
fragment of the clause that gives the rule its authority.

```python
RULE_PED = PolicyRule(
    rule_id="pre_existing_disease_48m",
    dimension=DecisionDimension.PRE_EXISTING_DISEASE,
    anchors=[Anchor(phrase="Pre-existing diseases will not be covered until 48 months")],
    applies=lambda ctx: ctx.case.treatment.pre_existing is True,
    evaluate=_eval_ped,
)
```

A rule may only fire if its anchor is found **inside a chunk that retrieval
actually returned for this case**. The consequences are structural rather than
advisory:

- **RULE 2** — a rule cannot assert a clause the policy does not contain,
  because the anchor would not resolve.
- **RULE 3** — citations are emitted *from* the resolved chunk, so page, section
  and `chunk_id` are copied from real retrieved evidence. There is no code path
  that constructs a page number.
- **RULE 4/5** — if retrieval misses the governing clause, the rule reports
  `UNRESOLVED` instead of guessing, which propagates to abstention.

**The failure mode this buys:** a retrieval regression degrades the system into
abstention, never into confident wrong answers. This is not theoretical — it is
exactly what happened in failure F-04, where a bad query change caused five
cases to abstain rather than to answer incorrectly.

Where a rule matches a term from one of the policy's own lists (the first-year
conditions, the exclusion lists), the matched term must appear in **both** the
claim and the retrieved clause text before the rule fires, so the citation
genuinely supports the sentence it is attached to.

---

## 3. Agent boundaries

Five agents, separated by *responsibility*, not by prompt. Each owns a distinct
capability and exchanges typed objects rather than prose.

```
ClaimCase
    │
    ▼
┌──────────────────────┐
│ 1. Case Analysis     │  facts, derived facts, decision dimensions,
│                      │  missing fields, irrelevant attributes,
│                      │  investigation plan, retrieval queries
└──────────┬───────────┘
           │ CaseAnalysis + list[RetrievalQuery]
           ▼
┌──────────────────────┐
│ 2. Policy Evidence   │  dense + BM25 → RRF fusion → rerank
│                      │  (owns the retrieval stack; interprets nothing)
└──────────┬───────────┘
           │ list[EvidenceItem] + RetrievalStats
           ▼
┌──────────────────────┐
│ 3. Coverage &        │  evidence-bound rule engine + limit calculators
│    Exclusion         │  (reports what the policy says; reaches no verdict)
└──────────┬───────────┘
           │ CoverageAssessment
           ▼
┌──────────────────────┐
│ 4. Decision          │  precedence logic → one of five statuses
│                      │  + measured confidence
└──────────┬───────────┘
           │ AnalysisResponse (draft)
           ▼
┌──────────────────────┐
│ 5. Validation        │  7 checks against retrieved evidence
│                      │  PASS → emit; FAIL → revise, then force NEEDS_REVIEW
└──────────┬───────────┘
           │ revision loop (max MAX_VALIDATION_REVISIONS)
           ▼
    AnalysisResponse
```

### Why these boundaries are real

The test that separation is meaningful is whether each agent could fail
independently and be tested independently. Here:

- **Case Analysis** decides *what to investigate*. Its output changes with the
  claim: a domiciliary claim pulls in the domiciliary dimension and its
  sub-limit; an 8-hour admission pulls in day care. Tested by asserting that
  dimension selection differs across claim types.
- **Policy Evidence** owns retrieval and performs **no** interpretation. This is
  what allows the coverage agent's conclusions to be checked against the
  evidence this agent returned — if the same component both retrieved and
  interpreted, validation would be checking its own work.
- **Coverage & Exclusion** interprets but reaches **no verdict**. It reports
  findings and limits. Keeping "what the policy says" separate from "what we
  therefore decide" is what makes the decision precedence auditable.
- **Decision** applies precedence and computes confidence. It never re-reads
  the policy.
- **Validation** re-derives support **independently** from the retrieved
  evidence, and can force the decision to change.

Each agent is unit-tested in isolation (`tests/test_agents.py`), including the
critical negative case: `test_rule_cannot_fire_without_its_clause` runs the
coverage agent with `evidence=[]` and asserts it produces `UNRESOLVED` findings
with zero citations rather than falling back on built-in knowledge.

---

## 4. Structured state

Agents exchange Pydantic models defined in `app/models/schemas.py`. The
LangGraph state object (`app/graph/state.py`) declares ownership explicitly:

| Key | Written by | Read by |
| --- | --- | --- |
| `case` | (input) | all |
| `analysis` | case_analysis | policy_evidence, coverage, decision |
| `queries` | case_analysis | policy_evidence |
| `evidence` | policy_evidence | coverage_exclusion, validation |
| `retrieval_stats` | policy_evidence | decision (confidence), trace |
| `assessment` | coverage_exclusion | decision, validation |
| `decision_draft` | decision | validation |
| `validation` | validation | decision (on revision) |
| `trace` | all (append-only reducer) | API response |

No agent passes free-form text to another. A `Finding` carries a typed
`FindingStatus`, a typed `Severity`, its `evidence_chunk_ids` and its
`Citation` objects — so the Decision Agent's precedence logic reads enum values,
not sentences.

---

## 5. Retrieval architecture

### 5.1 Structure-aware chunking

The policy is a legal instrument whose meaning lives in **atomic units**: a
single definition, a single numbered exclusion, a single sub-limit note. A
fixed-size window slices those in half and a retriever then returns the bottom
of exclusion 4 and the top of exclusion 5, leaving the reasoning layer to guess
which clause it is looking at.

The chunker therefore segments by document structure and applies size control
only *within* a unit:

1. Lines are tagged with their source page, so provenance survives.
2. Running headers and footers are detected empirically — a line appearing on
   ≥60% of pages is boilerplate — rather than by line index.
3. Top-level sections are detected from the wording's own headings.
4. Each section is segmented by the unit type it actually uses: definitions
   (`Term means …`), numbered exclusions, numbered benefits and NB notes,
   lettered claims-procedure blocks, numbered standard conditions.
5. Oversized units are split on sentence boundaries with overlap. Units are
   **never merged across a clause boundary**, because merging two exclusions
   would make a citation ambiguous.

**Result:** 137 chunks — 58 definitions, 21 numbered exclusions (matching the
policy exactly), 14 scope-of-cover units — with a 42–1564 character spread that
confirms the chunker is following structure rather than a size budget.

### 5.2 Hybrid retrieval

```
query ──┬─→ dense (cosine over L2-normalised embeddings) ──┐
        └─→ BM25 (Okapi, policy-tuned tokenizer) ──────────┴─→ RRF → rerank
```

**Why RRF rather than weighted score fusion.** BM25 scores are unbounded and
corpus-dependent; cosine similarities sit in [-1, 1]. Normalising them onto a
shared scale needs calibration that would have to be re-tuned whenever the
embedding backend changes — and this system has a *pluggable* embedding backend.
RRF consumes only ranks, so it is invariant to both. A small normalised-score
term breaks ties within a rank.

**Two-phase evidence selection** (see failure F-01):

- *Phase 1 — recall guarantee.* Each dimension is reranked against its own query
  and its best `PER_DIMENSION_K` chunks are reserved, so a narrow decisive
  dimension is not crowded out by a broad one.
- *Phase 2 — precision fill.* The remaining budget is filled from the
  multi-query fused pool, favouring chunks relevant across several dimensions.

**Anchor-directed retrieval** (see failure F-06). Broad topical queries find the
right *area* of the policy but cannot rank a short precise clause: a 43-character
exclusion matches only a fraction of a ten-topic sweep query. Since every rule
declares the clause it needs, `required_anchors_for_case()` turns those
declarations into targeted queries. Retrieval therefore runs in two registers —
topical and anchor-directed — which took `evidence_recall@k` from 93.2% to 100%.

### 5.3 Reranking

A reranker's defining property is that it scores the query and document
**together**, unlike a bi-encoder's independent embedding or BM25's
bag-of-words. Whether that function is a neural network is an implementation
choice.

The default `LexicalSemanticReranker` scores query–document interaction through
IDF-weighted term coverage, bigram adjacency, numeric-and-unit agreement
(decisive in a policy full of "48 months", "1.0%", "24 hours"), a section prior
derived from the decision dimension, and dense similarity as one input among
several. It is weaker than a trained cross-encoder on paraphrase and the README
says so — but it is deterministic, needs no download, and is reproducible in CI,
which is what makes the reported numbers trustworthy.

`RERANKER_BACKEND=cross_encoder` switches to a BGE cross-encoder where the
weights are reachable.

---

## 6. Decision precedence and abstention

```
1. blocking violation        → NOT_ADMISSIBLE
2. unresolved condition      → NEEDS_REVIEW
3. partially excluded portion→ PARTIALLY_ADMISSIBLE
4. binding monetary cap      → ADMISSIBLE_WITH_LIMITS
5. otherwise                 → ADMISSIBLE

then: affirmative decision with confidence < ABSTAIN_BELOW_CONFIDENCE
      → NEEDS_REVIEW
```

**Why blocking outranks abstention.** If cosmetic surgery is excluded, an
unresolved question about the facility's registration does not change the
outcome — the claim fails either way. Abstaining there would be noise, not
caution. Abstention is reserved for cases where the unresolved condition is
genuinely *decisive*.

**Why only affirmative decisions are downgraded on low confidence.** A rejection
already grounded in a cited blocking clause is not made safer by converting it
to a review; the evidence for it is on the page.

Abstention triggers, in order of how often they fire:

1. A condition precedent the evidence cannot establish (facility is not a
   Hospital, medical necessity unconfirmed, no domiciliary qualifying
   circumstance).
2. An anchor clause a relevant rule needs that retrieval did not return.
3. Confidence below threshold on an affirmative decision.
4. Validation failure that survives the revision budget.

An explicit `null` in `evidence_context` is treated as materially different from
an absent field: `null` means *the question was raised and could not be
established* → abstain; absent means *not in issue* → do not raise it. This
distinction is what separates PUB-006 (abstains) from PUB-001 (does not).

---

## 7. Validation

Seven checks, run against the retrieved evidence independently of the agents
that produced the draft:

| Check | Catches |
| --- | --- |
| `citation_integrity` | A citation naming a chunk that was never retrieved, or whose page/section drifted |
| `material_support` | An asserting finding with no citation |
| `unresolved_discipline` | An unresolved finding that nonetheless claims support |
| `quote_provenance` | A quote assembled rather than extracted from the cited chunk |
| `limit_arithmetic` | `allowed ≠ min(claimed, cap)` or a deduction that does not reconcile |
| `decision_alignment` | A rejection with no blocking finding; an abstention with no gap |
| `abstention_discipline` | An affirmative decision taken over an unresolved condition |

On failure the agent requests one revision; the Decision Agent re-runs with the
validation result folded into confidence. If the failure persists, the decision
is forced to `NEEDS_REVIEW` rather than shipped unverified.

`tests/test_validation.py` verifies these by **deliberately tampering** with a
known-good decision — fabricating a citation to page 99, rewriting a quote,
breaking limit arithmetic, flipping the decision — and asserting each is caught.

---

## 8. Confidence

Confidence measures **evidential support**, not model certainty:

```
raw   = 0.25·retrieval_quality + 0.25·evidence_coverage
      + 0.25·citation_support  + 0.25·decision_consistency

final = clamp(raw × validation_factor − missing_evidence_penalty, 0, 1)
```

`citation_support` sits near 1.0 by construction, because unsupported findings
cannot be produced. It is retained as a **regression alarm**: if it drops, the
evidence-binding guarantee has been broken somewhere. Full component
definitions are in `app/services/confidence.py` and README §14.

---

## 9. Principal trade-offs

| Decision | Chosen | Alternative | Why |
| --- | --- | --- | --- |
| Decision logic | Deterministic rules bound to evidence | LLM reasoning | Exact arithmetic, reproducible evaluation, auditable clause-to-statement links |
| LLM role | Optional, non-decisional | Central | Identical decisions with no key; no non-determinism in adjudication |
| Dense backend | Pluggable, LSA default | BGE only | Model hub unreachable on locked-down hosts and free tiers; system still works |
| Reranker | Pluggable, lexical default | BGE cross-encoder only | Deterministic and reproducible in CI; transformer used when available |
| BM25 | Implemented in-tree | `rank_bm25` | Tokenizer must keep "1.0%", "48", "24 hours"; per-term attribution needed |
| Vector index | numpy exact, FAISS optional | FAISS always | At 137 chunks exact search is one BLAS call and is exact by construction |
| Fusion | RRF | Weighted score sum | Scale-invariant, so it survives a backend swap without recalibration |
| Chunking | Structure-aware | Fixed-size windows | Citations must point at whole clauses |
| Ambiguous clause (F-02) | Abstain on the ambiguous branch | Pick a reading | The document does not settle it; a confident answer would be wrong half the time |

### The honest weakness

The offline defaults trade paraphrase generalisation for reproducibility. LSA
embeddings are fitted on a 137-chunk corpus, so they represent *this* policy's
vocabulary well and generalise poorly to phrasings it never uses. Three things
compensate: the BM25 arm handles exact legal terminology, anchor-directed
retrieval targets clauses directly rather than semantically, and evidence
binding converts any remaining retrieval miss into an abstention. Setting
`EMBEDDING_BACKEND=sentence_transformer` on a host that can reach the model hub
is a one-line change and the rest of the pipeline is unaffected.

---

## 10. What I would do next

1. **Expert review of the ground truth.** The 22 expected outcomes were derived
   from the wording by one person. An adjuster would likely disagree on at least
   the F-03 calibration.
2. **Adversarial retrieval tests.** Paraphrased claims that never use the
   policy's vocabulary would properly stress the LSA backend's weakness.
3. **Clause-level regression fixtures.** Pin each rule's anchor to a snapshot so
   a re-chunk that loses a clause fails loudly rather than degrading recall.
4. **Calibration on volume.** Confidence is currently validated only by the
   correct/incorrect separation on 22 cases, which is too small to calibrate a
   threshold with confidence.
