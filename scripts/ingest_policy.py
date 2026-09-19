"""Build the policy index.

    python -m scripts.ingest_policy [--force] [--inspect] [--query "text"]

Run once before starting the API. The index is written to ``data/index/`` and
is rebuilt automatically if the policy PDF changes (the manifest stores a
content hash).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.models.schemas import DecisionDimension, RetrievalQuery  # noqa: E402
from app.retrieval.index import HybridRetriever  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest and index the policy PDF.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if an index exists.")
    parser.add_argument("--inspect", action="store_true", help="Print the chunk breakdown.")
    parser.add_argument("--query", help="Run a test query against the built index.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    print(f"Policy : {settings.policy_pdf}")
    print(f"Index  : {settings.index_dir}")

    if not settings.policy_pdf.exists():
        print(f"\nERROR: policy PDF not found at {settings.policy_pdf}")
        print("Place the supplied policy there, or set POLICY_PDF_PATH.")
        return 1

    retriever = HybridRetriever(settings)
    started = time.perf_counter()

    if args.force:
        print("\nForcing a rebuild...")
        retriever.build(persist=True)
    elif retriever.load():
        print("\nLoaded the existing index (use --force to rebuild).")
    else:
        print("\nNo usable index found; building...")
        retriever.build(persist=True)

    elapsed = time.perf_counter() - started
    print(f"\nIndexed {len(retriever.chunks)} chunks in {elapsed:.1f}s")
    print("Backends: " + ", ".join(f"{k}={v}" for k, v in retriever.backend_info().items()))

    if settings.manifest_path.exists():
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
        print(f"Fingerprint: {manifest.get('policy_fingerprint')}")

    if args.inspect:
        print("\nChunks by section:")
        for section, count in Counter(c.section for c in retriever.chunks).most_common():
            print(f"  {count:4d}  {section}")
        lengths = sorted(c.char_count for c in retriever.chunks)
        print(
            f"\nChunk length: min={lengths[0]} median={lengths[len(lengths)//2]} "
            f"max={lengths[-1]} chars"
        )
        print(f"Pages indexed: {sorted({c.page for c in retriever.chunks})}")
        clause_refs = sorted({c.clause_ref for c in retriever.chunks if c.clause_ref})
        print(f"\nNumbered clauses isolated: {len(clause_refs)}")

    if args.query:
        print(f"\nTest query: {args.query!r}")
        evidence, stats = retriever.retrieve(
            [RetrievalQuery(
                dimension=DecisionDimension.SCOPE_OF_COVER,
                query=args.query,
                rationale="manual test query",
            )],
            top_k=5,
        )
        print(
            f"  dense={stats.dense_results} bm25={stats.bm25_results} "
            f"fused={stats.fused_results} reranked={stats.reranked_results} "
            f"({stats.elapsed_ms}ms)"
        )
        for item in evidence:
            print(f"\n  [{item.rerank_score:.3f}] p.{item.page} · {item.section} · {item.heading}")
            print(f"      {item.snippet(200)}")

    print("\nReady. Start the API with:  uvicorn app.api.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
