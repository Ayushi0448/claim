# Evaluation Results

_Generated 2026-09-19T06:51:35+00:00 · engine v1.0.0_

## Configuration

| Component | Backend |
| --- | --- |
| embedding | `lsa` |
| dense_index | `numpy` |
| sparse | `bm25` |
| reranker | `lexical_semantic` |
| chunks | `137` |
| orchestrator | `langgraph` |
| llm | `disabled` |
| engine_version | `1.0.0` |

## Headline metrics (all cases)

| Metric | Value |
| --- | --- |
| Cases evaluated | 22 (22 labelled) |
| **Decision accuracy** | **100.0%** |
| Deduction accuracy | 100.0% (11 cases with a monetary label) |
| Evidence recall@k | 100.0% |
| Citation hit rate | 100.0% |
| Citation correctness | 100.0% |
| Validation pass rate | 100.0% |
| Unsupported claims detected | 0 |
| Mean latency | 264.0 ms (median 248.5 ms, max 572 ms) |

## Abstention performance

| Metric | Value |
| --- | --- |
| Cases where NEEDS_REVIEW is correct | 4 |
| Abstentions produced | 4 |
| Correct abstentions | 4 |
| Missed abstentions (decided when it should not have) | 0 |
| Over-abstentions (abstained when a decision was available) | 0 |
| Precision / Recall / F1 | 100.0% / 100.0% / 100.0% |

## Confidence calibration

- Mean confidence overall: **0.9592**
- Mean confidence when the decision was correct: **0.9592**
- Mean confidence when the decision was incorrect: **n/a (no incorrect decisions)**

## By suite

| Suite | Cases | Decision accuracy | Evidence recall@k | Citation correctness | Validation pass |
| --- | --- | --- | --- | --- | --- |
| public | 12 | 100.0% | 100.0% | 100.0% | 100.0% |
| custom | 10 | 100.0% | 100.0% | 100.0% | 100.0% |

## Accuracy by expected decision

| Expected decision | Cases | Correct | Accuracy |
| --- | --- | --- | --- |
| ADMISSIBLE | 5 | 5 | 100.0% |
| ADMISSIBLE_WITH_LIMITS | 5 | 5 | 100.0% |
| NEEDS_REVIEW | 4 | 4 | 100.0% |
| NOT_ADMISSIBLE | 7 | 7 | 100.0% |
| PARTIALLY_ADMISSIBLE | 1 | 1 | 100.0% |

## Per-case results

| Case | Suite | Expected | Actual | OK | Conf | Recall@k | Cite ✓ | Val | ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUB-001 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 0.97 | 1.0 | 1.0 | PASS | 572 |
| PUB-002 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 385 |
| PUB-003 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 0.97 | 1.0 | 1.0 | PASS | 282 |
| PUB-004 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 1.00 | 1.0 | 1.0 | PASS | 169 |
| PUB-005 | public | ADMISSIBLE | ADMISSIBLE | ✅ | 0.97 | 1.0 | 1.0 | PASS | 264 |
| PUB-006 | public | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.86 | 1.0 | 1.0 | PASS | 266 |
| PUB-007 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 0.97 | 1.0 | 1.0 | PASS | 231 |
| PUB-008 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 220 |
| PUB-009 | public | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 303 |
| PUB-010 | public | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 284 |
| PUB-011 | public | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.89 | 1.0 | 1.0 | PASS | 230 |
| PUB-012 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 250 |
| CUS-001 | custom | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 225 |
| CUS-002 | custom | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 1.00 | 1.0 | 1.0 | PASS | 258 |
| CUS-003 | custom | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 217 |
| CUS-004 | custom | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 247 |
| CUS-005 | custom | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.70 | 1.0 | 1.0 | PASS | 259 |
| CUS-006 | custom | PARTIALLY_ADMISSIBLE | PARTIALLY_ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 246 |
| CUS-007 | custom | NOT_ADMISSIBLE | NOT_ADMISSIBLE | ✅ | 0.97 | 1.0 | 1.0 | PASS | 298 |
| CUS-008 | custom | ADMISSIBLE | ADMISSIBLE | ✅ | 1.00 | 1.0 | 1.0 | PASS | 230 |
| CUS-009 | custom | NEEDS_REVIEW | NEEDS_REVIEW | ✅ | 0.89 | 1.0 | 1.0 | PASS | 165 |
| CUS-010 | custom | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ✅ | 1.00 | 1.0 | 1.0 | PASS | 206 |

## Failures

No decision mismatches against the ground truth in this run.

## Retrieval gaps

Every required policy clause was retrieved for every case.

---

Reproduce with:

```bash
python -m evaluation.run_evaluation
```
