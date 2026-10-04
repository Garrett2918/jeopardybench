"""Offline stand-ins for players and the judge (`run.py --mock`, tests, viewer demos)."""
from __future__ import annotations

import random

from .log import Verdict
from .players import Outcome

QUIPS = ["Easy money.", "I have read this paper. Twice.", "Feeling lucky.", "Pass the chalk.",
         "This one is mine.", "Bold of the writers.", "Hmm.", "", "", ""]


def normalise(response: str) -> str:
    r = response.strip().rstrip("?.!").strip()
    for prefix in ("what is ", "what are ", "who is ", "who are "):
        if r.lower().startswith(prefix):
            r = r[len(prefix):]
    return r.strip().lower()


class MockPlayer:
    def __init__(self, name: str, model: str, color: str, answers: dict[str, str], skill: float,
                 seed: int, native_reasoning: bool = True, failure_rate: float = 0.03):
        self.name, self.model, self.color = name, model, color
        self.answers, self.skill = answers, skill
        self.rng = random.Random(seed)
        self.native_reasoning = native_reasoning
        self.failure_rate = failure_rate
        self.usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "failures": 0}

    def _outcome(self, data: dict | None, reasoning: str | None = None) -> Outcome:
        self.usage["calls"] += 1
        t_in, t_out = self.rng.randint(200, 900), self.rng.randint(100, 8000)
        self.usage["tokens_in"] += t_in
        self.usage["tokens_out"] += t_out
        if data is None:
            self.usage["failures"] += 1
        return Outcome(
            data=data, error=None if data else "timeout", latency_ms=self.rng.randint(800, 90000),
            tokens={"in": t_in, "out": t_out},
            reasoning={"kind": "native" if self.native_reasoning else "summary", "text": reasoning} if reasoning else None,
            served_model=self.model,
        )

    def _answer(self, ctx: dict, forced: bool) -> Outcome:
        if self.rng.random() < self.failure_rate:
            return self._outcome(None)
        truth = self.answers[ctx["qid"]]
        right = self.rng.random() < self.skill
        guess = truth if right else self.rng.choice(["A", "B", "17", "0", "42", "The Krebs cycle"])
        if not right and normalise(guess) == normalise(truth):
            guess = "None of the above"
        mean = 72 if right else 38
        conf = max(0, min(100, int(self.rng.gauss(mean, 18))))
        reasoning = (f"Let me think about {ctx['subject']}. The clue asks: {ctx['question'][:120]}...\n"
                     f"Working it through, I get {guess}. Check: consistent. Confidence about {conf}%. "
                     "Edge case: what if $x<0$? Excluded by the clue. <script>alert(1)</script>")
        return self._outcome({
            "response": f"What is {guess}?",
            "explanation": f"Mock reasoning for {ctx['subject']}: the answer follows from the setup.",
            "confidence": conf,
            "buzz": True if forced else conf >= 55,
            "quip": self.rng.choice(QUIPS),
        }, reasoning)

    def answer_clue(self, ctx: dict) -> Outcome:
        return self._answer(ctx, forced=False)

    def answer_forced(self, ctx: dict, kind: str) -> Outcome:
        return self._answer(ctx, forced=True)

    def pick(self, ctx: dict) -> Outcome:
        subject = self.rng.choice(sorted(ctx["remaining"]))
        return self._outcome({"subject": subject, "value": self.rng.choice(ctx["remaining"][subject])})

    def wager(self, ctx: dict, kind: str, low: int, high: int) -> Outcome:
        return self._outcome({"wager": self.rng.randint(low, high), "quip": self.rng.choice(QUIPS)})

    def preflight(self) -> Outcome:
        return self._outcome({"ok": True, "answer": "Paris"})


class MockJudge:
    model = "mock-judge"

    def __init__(self):
        self.usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "failures": 0}

    def grade(self, question: str, answer: str, response: str, answer_type: str = "exactMatch") -> Verdict:
        self.usage["calls"] += 1
        got = normalise(response)
        ok = bool(got) and got == normalise(answer)
        return Verdict(ok, got, "Exact match." if ok else f"'{got}' does not match '{answer}'.")
