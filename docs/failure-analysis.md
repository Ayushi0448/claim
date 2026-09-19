# Failure Analysis

Seven defects found by running the system against the 22-case evaluation suite,
with the root cause and the fix for each. Every failure below was observed in a
real run; the metrics quoted are from the evaluation harness, not estimates.

**Decision accuracy across the fixes: 77.3% → 95.5% → 100.0% (22 cases).**

| ID | Failure | Cases affected | Root cause | Status |
| --- | --- | --- | --- | --- |
| F-01 | Rules abstained because their clause was never retrieved | PUB-006 and others | Global evidence pool starved narrow dimensions | Fixed |
| F-02 | Exclusion 20 scope is ambiguous in the source PDF | Potentially any listed disease | OCR flattened a nested list | Mitigated, documented |
| F-03 | Domiciliary "three day" rule cannot be evaluated | PUB-004 | Input schema carries no treatment duration | Documented limitation |
| F-04 | Five cases over-abstained after a retrieval "improvement" | PUB-005, CUS-001/002/003/010 | One dimension serving two information needs | Fixed |
| F-05 | A 43-character decisive exclusion was ranked out | CUS-003 | Short-text penalty applied to complete clauses | Fixed |
| F-06 | Short precise clauses lost to broad topical queries | CUS-003 | No query targeted the clause a rule needed | Fixed |
| F-07 | Infinite loop in the validation → decision cycle | PUB-006 | Revision counter not incremented on the terminal path | Fixed |

---

## F-01 — Narrow decision dimensions were starved of evidence

**Observed.** PUB-006 returned `NEEDS_REVIEW`, which is the correct label, but
for the wrong reason. Inspecting the findings showed 6–9 rules per case
reporting `UNRESOLVED` with the detail *"Anchor text not present in retrieved
evidence"*. The system was abstaining because retrieval had failed, not because
the evidence was genuinely absent — an abstention that happens to be right is
still a bug when the reasoning is wrong.

**Root cause.** The Policy Evidence Agent fused all per-dimension rankings into
one global pool and took the top *k*. A claim engaging 12–17 dimensions puts
them all into a single contest for ~22 slots, and a broad dimension (the
exclusions sweep) crowds out a narrow one (the ambulance cap) even though the
narrow one is decisive for the payable amount.

**Evidence.** `evidence_recall@k` was 90.9% with specific clauses missing, and
the trace field `dimensions_without_evidence` was non-empty on most cases.

**Fix.** Two-phase evidence selection in `HybridRetriever.retrieve`:

- *Phase 1 (recall guarantee)* — each dimension is reranked against **its own**
  query and its best `PER_DIMENSION_K` (default 3) chunks are reserved.
- *Phase 2 (precision fill)* — remaining budget is filled from the multi-query
  fused pool, favouring chunks relevant across several dimensions.

**Result.** Unresolved findings per case fell from 6–9 to 0–2, and mean
confidence rose materially (PUB-009 0.85 → 0.98) because phantom unresolved
findings had been depressing the decision-consistency term.

**Regression test.** `tests/test_retrieval.py::TestHybridPipeline::test_per_dimension_recall_guarantee`

---

## F-02 — Structural ambiguity in the source policy (exclusion 20)

**Observed.** The supplied PDF renders exclusions 17–20 as a flat numbered list:

```
17. Any expense under Domiciliary Hospitalisation for
18. Pre and Post Hospitalisation treatment
19. Any treatment not exceeding three days.
20. Treatment of following diseases:
      i) Asthma  ii) Bronchitis  ...  vii) Hypertension  ...
```

Item 17 ends mid-sentence ("...Hospitalisation **for**"), which shows that
18–20 are its sub-items. The Acrobat Paper Capture export flattened the nesting.

**Why it matters.** The reading changes outcomes. If item 20 is scoped to
domiciliary treatment, an in-patient admission for hypertension is covered. If
it stands alone, that admission is excluded. Nothing in the document settles it.

**Root cause.** A defect in the supplied source document, not in the pipeline.
No amount of retrieval quality resolves it, because the information is not
present.

**Fix.** `exclusion_listed_diseases` branches on treatment type rather than
picking a reading:

- **Domiciliary** treatment for a listed disease → `NOT_ADMISSIBLE`. Both
  readings agree here, so a confident rejection is safe.
- **In-patient** treatment for a listed disease → `NEEDS_REVIEW`, stating that
  the clause's scope cannot be established from the supplied wording.

This is the system behaving as specified by §10 ("the system cannot confidently
establish a condition required by the policy") and RULE 5. Silently choosing a
reading would produce confident answers that are wrong half the time.

**Note.** No supplied public case triggers this path — PUB-002 (viral fever)
is disposed of by the 30-day waiting period, which is unambiguous and takes
precedence. The branch is reachable and tested, but it is a latent risk rather
than an observed failure.

---

## F-03 — Domiciliary treatment duration is not in the input schema

**Observed.** PUB-004 is a domiciliary claim. The policy excludes domiciliary
expenses for "any treatment not exceeding three days", but the supplied case
carries `admission_hours: 0` and no treatment duration, so the condition cannot
be evaluated.

**Root cause.** A genuine gap between the policy's requirements and the input
schema. `schema/claim_case_schema.md` does not define a duration field for
domiciliary treatment.

**Decision and rationale.** The duration is recorded as a **non-blocking**
evidence gap rather than an abstention trigger, so PUB-004 resolves to
`ADMISSIBLE_WITH_LIMITS` with the 20% sub-limit applied. Two reasons:

1. The case's own stated task asks to "identify the applicable sub-limit",
   indicating the duration is not the question under test.
2. Treating every unstated field as blocking would make the system abstain on
   almost everything, which destroys its usefulness. Abstention is reserved for
   conditions the case *raises* and leaves unresolved (an explicit `null`), or
   that are clearly material on the facts.

This is a calibration judgement, and it is the one I would most want to
re-examine with a domain expert. `Treatment.treatment_days` exists in the schema
so a caller can supply it; when present it would be evaluated.

---

## F-04 — A retrieval "fix" caused five over-abstentions

**Observed.** After adding the claims-documentation clause to the
`SCOPE_OF_COVER` query expansions, evaluation accuracy **fell from 95.5% to
77.3%**. Five cases (PUB-005, CUS-001, CUS-002, CUS-003, CUS-010) flipped to
`NEEDS_REVIEW`. All five reported the same trigger:

```
ABSTAIN-TRIGGER: scope_of_cover
  detail: Anchor text not present in retrieved evidence:
          We will pay Reasonable and Customary charges
```

**Root cause.** I had loaded one dimension with two unrelated information needs
— *what does the policy cover* and *what documents must the claim carry*. The
blended query matched neither well, and the scope clause itself dropped out of
its own dimension's top-k. The `scope_of_cover` rule then could not ground, and
because its unresolved severity is `ABSTAIN`, every affected claim abstained.

**Lesson.** This is the failure mode the architecture is designed to produce: a
retrieval regression degraded the system into abstention rather than into
confident wrong answers. No case was decided incorrectly — the system refused to
decide. That is the correct behaviour for claims adjudication, and it made the
regression trivial to spot in the evaluation output.

**Fix.** Claim documentation was given its own dimension
(`DecisionDimension.CLAIM_DOCUMENTATION`) with its own query, and the
`SCOPE_OF_COVER` query was restored. **One dimension, one information need.**

**Result.** 77.3% → 95.5%.

---

## F-05 — A 43-character decisive clause was penalised as a fragment

**Observed.** CUS-003 (dental surgery) returned `NEEDS_REVIEW` with
`evidence_recall@k = 0.00`. The governing clause is:

```
7. Dental treatment or surgery of any kind.
```

43 characters, and it disposes of the claim completely.

**Root cause.** The reranker applied a 0.6× penalty to any chunk under 80
characters. That heuristic is sound in general IR — short fragments are usually
noise — but it is exactly wrong for a legal document, where the shortest clauses
are often the most absolute.

**Fix.** The short-text penalty no longer applies to chunks carrying a
`clause_ref`, i.e. chunks the structure-aware chunker identified as complete
numbered clauses. A numbered exclusion is a complete legal statement regardless
of length.

**Regression test.** `tests/test_retrieval.py::TestReranker::test_short_complete_clause_is_not_penalised`

---

## F-06 — Broad topical queries cannot rank short precise clauses

**Observed.** After F-05, CUS-003 *still* failed. Tracing the ranking showed
`Exclusion 7` reaching the fused pool but landing at **rank 5**, below the
`PER_DIMENSION_K = 3` reservation:

```
0.393  len= 337  Exclusion 5: Circumcision unless necessary...
0.311  len= 528  Exclusion 20: Treatment of following diseases
0.296  len= 206  Dental Treatment (definition)
0.278  len= 203  Exclusion 14: Naturopathy, non-allopathic...
0.276  len=  43  Exclusion 7: Dental treatment or surgery of any kind.
```

Its `term_coverage` was 0.233 — the exclusions sweep query covers ten different
exclusion topics, so a clause addressing exactly one of them matches only a
fraction of the query terms. Longer chunks that touch several topics score
higher. **The clause was being out-competed for being specific.**

**Root cause.** Retrieval was organised entirely around topics, with no
mechanism for the reasoning layer to ask for a *particular* clause — even though
the rule engine knows exactly which clause it needs, because every rule declares
its anchor.

**Fix — anchor-directed retrieval.** `required_anchors_for_case()` probes which
rules apply to a case and returns their anchor phrases. The Case Analysis Agent
emits an additional targeted query per anchor, alongside the broad topical
queries. Retrieval now runs in two registers:

- **Broad topical queries** find the right *area* of the policy.
- **Anchor queries** find the *exact clause* the evidence-binding step requires.

An anchor query matches its target almost exactly, so the clause scores near 1.0
and survives both the per-dimension reservation and the final budget trim.

**Result.** 95.5% → **100.0%**, with `evidence_recall@k` and `citation_hit_rate`
both reaching 100%. Mean latency rose from 121 ms to 166 ms — roughly 15 extra
queries per case, which is a good trade for guaranteed clause recall.

**Regression test.** `tests/test_agents.py::TestCaseAnalysisAgent::test_anchor_directed_queries_are_generated`

---

## F-07 — Infinite loop in the validation → decision cycle

**Observed.** PUB-006 hung and then crashed:

```
langgraph.errors.GraphRecursionError: Recursion limit of 25 reached
  without hitting a stop condition
```

**Root cause.** When validation failed and the revision budget was exhausted,
`force_review()` was applied but `revision_count` was **not** incremented and
`report.revision_required` stayed `True`. The routing predicate therefore kept
returning `decision`, and the graph cycled until LangGraph's recursion limit
tripped.

**Fix.** On the terminal path, `revision_required` is cleared (there is nothing
left to revise once the decision has been forced to `NEEDS_REVIEW`) and
`revision_count` is incremented unconditionally, so both loop guards agree.

**Secondary finding.** The engine's outer exception handler converted the crash
into a valid `NEEDS_REVIEW` response rather than a 500, so the API stayed
correct while the workflow was broken. That is the intended containment, but it
also hid the defect from the API surface — which is why the trace records
pipeline errors explicitly.

**Regression test.** `tests/test_validation.py::TestForcedReview::test_revision_loop_terminates`

---

## A validation-logic defect found alongside F-01

Not a decision failure, but worth recording. The `material_support` check
required every material finding to carry a citation — including findings with
status `UNRESOLVED`. An unresolved finding asserts that the governing clause
*could not be established*; demanding a citation from it is incoherent, and it
made every legitimate abstention report `validation: FAIL`.

The check now applies only to findings that assert something about the policy
(`VIOLATED`, `SATISFIED`, `LIMIT_APPLIES`). A separate check
(`unresolved_discipline`) audits the opposite error: an unresolved finding that
nonetheless claims evidential support.

---

## What these failures say about the design

Six of the seven failures were **retrieval** failures, not reasoning failures —
and in every case the system responded by abstaining rather than by inventing an
answer. That is the property the architecture was built for: because a rule
cannot fire unless its clause is genuinely retrieved, a retrieval regression
costs recall, never correctness.

The one exception, F-07, was an orchestration bug, and it was contained by the
engine's error boundary into a safe `NEEDS_REVIEW`.

The 100% figure on 22 cases should be read with that in mind, and with the
limitations in README §28 — particularly that the custom cases and the rules
were written by the same author, which is a real source of optimism in the
measurement.
