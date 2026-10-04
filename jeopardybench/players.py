"""Contestants over an OpenAI-compatible API. Spec §5.6, §6.1.

A Player turns game contexts into prompts, calls its Transport, validates the JSON reply
(retrying once on a bad reply) and returns an Outcome. Failures never raise: the Outcome
carries `error` and `data=None`, and the game applies the fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import prompts as P


@dataclass
class Completion:
    content: str | None
    reasoning: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int | None = None
    latency_ms: int = 0
    served_model: str | None = None
    finish_reason: str | None = None


class CallError(Exception):
    """Timeout, HTTP error or connection failure (after any transport-level retry)."""


@dataclass
class Outcome:
    data: dict | None
    error: str | None = None
    latency_ms: int = 0
    tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    reasoning_tokens: int | None = None
    reasoning: dict | None = None  # {"kind": "native" | "summary", "text": str}
    served_model: str | None = None


class OpenAITransport:
    def __init__(self, base_url: str, api_key: str, timeout_s: float, sleep: Callable = time.sleep):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.sleep = sleep

    def complete(self, model: str, messages: list[dict], params: dict) -> Completion:
        o = self.openai
        for attempt in range(2):
            start = time.monotonic()
            try:
                r = self.client.chat.completions.create(model=model, messages=messages, extra_body=params or None)
                break
            except o.APITimeoutError:
                raise CallError("timeout") from None
            except (o.RateLimitError, o.InternalServerError, o.APIConnectionError) as e:
                if attempt == 0:
                    self.sleep(5)
                    continue
                raise CallError(f"{type(e).__name__}: {str(e)[:200]}") from None
            except o.APIStatusError as e:
                raise CallError(f"HTTP {e.status_code}: {str(e)[:200]}") from None
        latency = int((time.monotonic() - start) * 1000)
        if not r.choices:
            raise CallError("no choices in response")
        msg = r.choices[0].message.model_dump()
        usage = r.usage.model_dump() if r.usage else {}
        details = usage.get("completion_tokens_details") or {}
        return Completion(
            content=msg.get("content"),
            reasoning=extract_reasoning(msg),
            tokens_in=usage.get("prompt_tokens") or 0,
            tokens_out=usage.get("completion_tokens") or 0,
            reasoning_tokens=details.get("reasoning_tokens"),
            latency_ms=latency,
            served_model=r.model,
            finish_reason=r.choices[0].finish_reason,
        )


def extract_reasoning(msg: dict) -> str | None:
    for key in ("reasoning_content", "reasoning"):
        if isinstance(msg.get(key), str) and msg[key].strip():
            return msg[key]
    details = msg.get("reasoning_details")
    if isinstance(details, list):
        parts = [d.get("text") or d.get("summary") or "" for d in details if isinstance(d, dict)]
        text = "\n".join(p for p in parts if isinstance(p, str) and p)
        return text or None
    return None


class Player:
    def __init__(self, name: str, model: str, color: str, transport, params: dict | None = None,
                 native_reasoning: bool = False):
        self.name, self.model, self.color = name, model, color
        self.transport = transport
        self.params = params or {}
        self.native_reasoning = native_reasoning
        self.usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "failures": 0}

    # -- game-facing API --------------------------------------------------
    def answer_clue(self, ctx: dict) -> Outcome:
        return self.call(P.clue(ctx), P.parse_clue, P.CLUE_SCHEMA)

    def answer_forced(self, ctx: dict, kind: str) -> Outcome:
        return self.call(P.forced_answer(ctx, kind), lambda t: P.parse_clue(t, forced=True), P.FORCED_SCHEMA)

    def pick(self, ctx: dict) -> Outcome:
        return self.call(P.pick(ctx), P.parse_pick, '{"subject": "...", "value": 400}')

    def wager(self, ctx: dict, kind: str, low: int, high: int) -> Outcome:
        return self.call(P.wager(ctx, kind), lambda t: P.parse_wager(t, low, high), '{"wager": 1000}')

    def preflight(self) -> Outcome:
        return self.call(P.preflight(), P.parse_preflight, '{"ok": true, "answer": "..."}')

    # -- plumbing -----------------------------------------------------------
    def call(self, messages: list[dict], parse: Callable[[str], dict], schema: str) -> Outcome:
        out = Outcome(data=None)
        for attempt in range(2):
            try:
                c = self.transport.complete(self.model, messages, self.params)
            except CallError as e:
                out.error = str(e)
                break
            self.usage["calls"] += 1
            self.usage["tokens_in"] += c.tokens_in
            self.usage["tokens_out"] += c.tokens_out
            out.latency_ms += c.latency_ms
            out.tokens["in"] += c.tokens_in
            out.tokens["out"] += c.tokens_out
            out.reasoning_tokens = c.reasoning_tokens
            out.served_model = c.served_model
            if c.reasoning:
                out.reasoning = {"kind": "native" if self.native_reasoning else "summary", "text": c.reasoning}
            try:
                out.data, out.error = parse(c.content), None
                break
            except ValueError as e:
                if c.finish_reason == "length":
                    # Out of tokens; a retry with the same budget would just repeat it.
                    out.error = f"ran out of tokens (max_tokens={self.params.get('max_tokens', 'default')})"
                    break
                out.error = f"bad reply: {e}"
                messages = messages + [
                    {"role": "assistant", "content": c.content or ""},
                    {"role": "user", "content": f"Reply with JSON only, matching exactly: {schema}"},
                ]
        if out.error:
            self.usage["failures"] += 1
        return out
