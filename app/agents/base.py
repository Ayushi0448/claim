"""Shared agent scaffolding: timing and trace emission."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import TraceEvent


@dataclass
class AgentTimer:
    """Collects one trace event per agent run.

    Only concise, auditable facts are recorded — agent name, action taken,
    counts, status and elapsed time. No reasoning text is ever emitted, which
    is how RULE 10 (do not expose chain-of-thought) is enforced structurally
    rather than by convention.
    """

    agent: str
    action: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    _started: float = 0.0
    elapsed_ms: int = 0

    def __enter__(self) -> "AgentTimer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._started) * 1000)

    def event(self) -> TraceEvent:
        return TraceEvent(
            agent=self.agent,
            action=self.action,
            elapsed_ms=self.elapsed_ms,
            status=self.status,
            metrics=self.metrics,
        )


@contextmanager
def trace_step(agent: str, action: str):
    timer = AgentTimer(agent=agent, action=action)
    with timer:
        yield timer
