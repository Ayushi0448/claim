"""Optional LLM client.

Deliberate scope limit
----------------------
The LLM in this system does **not** decide admissibility. Admissibility is a
deterministic function of the policy clauses and the claim facts, computed by
the rule engine, where every statement is bound to a retrieved clause.

The LLM is given three narrow jobs, all of them optional:

1. ``expand_queries``  — propose extra retrieval phrasings for a dimension.
   Failure mode: fewer query expansions. Harmless.
2. ``summarise``       — write the reviewer-facing prose summary.
   Failure mode: a deterministic template summary is used instead.
3. ``review_claims``   — an independent second opinion during validation,
   asked whether a statement is supported by the quoted evidence.
   Failure mode: the deterministic validator runs alone.

Because none of these can change a decision, running with ``LLM_PROVIDER=none``
produces identical decisions to running with a key — which is what makes the
evaluation reproducible. Any provider exposing an OpenAI-compatible
``/chat/completions`` endpoint works (OpenRouter, Groq, Together, Ollama).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import ModelSettings, get_settings

logger = logging.getLogger(__name__)

_OPENAI_COMPATIBLE_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
}


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    error: str | None = None
    elapsed_ms: int = 0


class LLMClient:
    """Thin, dependency-light OpenAI-compatible chat client."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.cfg = settings or get_settings().models

    @property
    def enabled(self) -> bool:
        return self.cfg.llm_enabled

    @property
    def description(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"{self.cfg.llm_provider}:{self.cfg.llm_model or 'default'}"

    def _base_url(self) -> str:
        if self.cfg.llm_base_url:
            return self.cfg.llm_base_url.rstrip("/")
        return _OPENAI_COMPATIBLE_BASE_URLS.get(
            self.cfg.llm_provider, "https://api.openai.com/v1"
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResult:
        """Single chat completion. Never raises; failures degrade gracefully."""
        if not self.enabled:
            return LLMResult(ok=False, error="LLM disabled (set LLM_PROVIDER and LLM_API_KEY)")

        import time

        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.cfg.llm_model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.llm_api_key}"

        last_error = "unknown error"
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                import httpx

                with httpx.Client(timeout=self.cfg.llm_timeout_s) as client:
                    response = client.post(
                        f"{self._base_url()}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    continue
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return LLMResult(
                    ok=True,
                    content=content,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.debug("LLM attempt %d failed: %s", attempt + 1, last_error)

        logger.info("LLM call failed, continuing without it: %s", last_error)
        return LLMResult(
            ok=False, error=last_error, elapsed_ms=int((time.perf_counter() - started) * 1000)
        )

    # ------------------------------------------------------------------ #

    def complete_json(self, system: str, user: str, **kwargs: Any) -> dict | list | None:
        """Completion parsed as JSON, tolerating fenced output."""
        result = self.complete(system, user, **kwargs)
        if not result.ok:
            return None
        text = result.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = min(
                (i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1
            )
            end = max(text.rfind("}"), text.rfind("]"))
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None


_CLIENT: LLMClient | None = None


def get_llm_client(settings: ModelSettings | None = None) -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient(settings)
    return _CLIENT
