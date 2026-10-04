"""The game state machine. Spec §5.

Players and judge are injected (real or mock). All randomness goes through `rng`.
Every clue is answered privately by every player and every reply is judged; `counted`
marks the replies that actually affected the score.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Callable

import pandas as pd

from .board import GameBoard
from .log import Cell, Reply


class Game:
    def __init__(self, board: GameBoard, questions: pd.DataFrame, players: list, judge,
                 rng: random.Random, on_event: Callable[[dict], None] | None = None):
        self.board, self.q, self.players, self.judge, self.rng = board, questions, players, judge, rng
        self.on_event = on_event or (lambda e: None)
        self.names = [p.name for p in players]
        self.scores = [0] * len(players)
        self.control = 0
        self.events: list[dict] = []
        self.reasoning: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=max(4, 2 * len(players)))

    # -- top level ----------------------------------------------------------
    def play(self) -> dict:
        try:
            for r_i, rnd in enumerate(self.board.rounds):
                self.play_round(r_i)
            self.play_final()
        finally:
            self.pool.shutdown(wait=False)
        best = max(self.scores)
        return {"scores": list(self.scores), "winners": [i for i, s in enumerate(self.scores) if s == best]}

    def play_round(self, r_i: int) -> None:
        if r_i == 0:
            self.control = self.rng.randrange(len(self.players))
        else:
            low = min(self.scores)
            self.control = self.rng.choice([i for i, s in enumerate(self.scores) if s == low])
        self.emit({"type": "round_start", "round": r_i, "control": self.control, "scores": list(self.scores)})
        remaining = list(self.board.rounds[r_i].cells)
        while remaining:
            cell = self.pick(r_i, remaining)
            remaining.remove(cell)
            if cell.daily_double:
                self.daily_double(r_i, cell)
            else:
                self.regular_clue(r_i, cell)

    # -- picking --------------------------------------------------------------
    def pick(self, r_i: int, remaining: list[Cell]) -> Cell:
        rnd = self.board.rounds[r_i]
        options: dict[str, list[int]] = {}
        for c in remaining:
            options.setdefault(c.subject, []).append(c.value)
        options = {s: sorted(v) for s, v in options.items()}
        player = self.players[self.control]
        out = player.pick({**self.ctx(r_i), "name": player.name, "remaining": options})
        cell = None
        if out.data:
            subject = next((s for s in options if s.lower() == out.data["subject"].lower()), None)
            cell = next((c for c in remaining if c.subject == subject and c.value == out.data["value"]), None)
        fallback = cell is None
        if fallback:
            subject = self.rng.choice(sorted(options))
            cell = next(c for c in remaining if c.subject == subject and c.value == options[subject][0])
        self.emit({"type": "pick", "round": r_i, "player": self.control, "subject": cell.subject,
                   "value": cell.value, "row": cell.row, "col": cell.col, "fallback": fallback,
                   "error": (out.error or "invalid pick") if fallback else None})
        return cell

    # -- clues ----------------------------------------------------------------
    def regular_clue(self, r_i: int, cell: Cell) -> None:
        ctx = {**self.ctx(r_i), **self.clue_ctx(cell)}
        replies, reasoning = self.ask_all(lambda p: p.answer_clue({**ctx, "name": p.name}))
        self.judge_all(cell.qid, replies)
        order = sorted((r for r in replies if r.buzz), key=lambda r: (-r.confidence, r.latency_ms))
        for r in order:
            r.counted = True
            if r.verdict.correct:
                self.scores[r.player] += cell.value
                self.control = r.player
                break
            self.scores[r.player] -= cell.value
        self.emit_clue(r_i, cell, replies, reasoning, buzz_order=[r.player for r in order], wager=None)

    def daily_double(self, r_i: int, cell: Cell) -> None:
        p_i = self.control
        player = self.players[p_i]
        rnd = self.board.rounds[r_i]
        high = max(self.scores[p_i], max(rnd.values))
        base = {**self.ctx(r_i), "subject": cell.subject}
        w = player.wager({**base, "name": player.name, "max_wager": high}, "dd", 5, high)
        wager = w.data["wager"] if w.data else 5
        ctx = {**base, **self.clue_ctx(cell), "wager": wager}
        replies, reasoning = self.ask_all(
            lambda p: p.answer_forced({**ctx, "name": p.name}, "dd") if p is player
            else p.answer_clue({**ctx, "name": p.name, "wager": None}))
        self.judge_all(cell.qid, replies)
        r = replies[p_i]
        r.counted = r.buzz = True
        self.scores[p_i] += wager if r.verdict.correct else -wager
        self.emit_clue(r_i, cell, replies, reasoning, buzz_order=[p_i], wager=wager,
                       dd={"player": p_i, "max_wager": high, "quip": w.data["quip"] if w.data else "",
                           "error": w.error})

    def play_final(self) -> None:
        final = self.board.final
        qualifiers = [i for i, s in enumerate(self.scores) if s > 0]
        before = list(self.scores)
        if not qualifiers:
            self.emit({"type": "final", "skipped": True, "subject": final["subject"], "qid": final["qid"],
                       "scores_before": before, "scores_after": list(self.scores)})
            return
        base = {"names": self.names, "scores": before, "round": "Final Jeopardy", "subject": final["subject"]}
        wagers: list[int | None] = [None] * len(self.players)
        wager_info: list[dict | None] = [None] * len(self.players)
        futures = {i: self.pool.submit(self.players[i].wager,
                                       {**base, "name": self.players[i].name, "max_wager": before[i]},
                                       "final", 0, before[i]) for i in qualifiers}
        for i, f in futures.items():
            w = f.result()
            wagers[i] = w.data["wager"] if w.data else 0
            wager_info[i] = {"quip": w.data["quip"] if w.data else "", "error": w.error}
        question = self.q.at[final["qid"], "question"]
        ctx = {**base, "question": question, "qid": final["qid"], "value": 0}
        replies, reasoning = self.ask_all(
            lambda p: p.answer_forced({**ctx, "name": p.name, "wager": wagers[self.players.index(p)]}, "final")
            if self.players.index(p) in qualifiers else p.answer_clue({**ctx, "name": p.name}))
        self.judge_all(final["qid"], replies)
        for i in qualifiers:
            r = replies[i]
            r.counted = r.buzz = True
            self.scores[i] += wagers[i] if r.verdict.correct else -wagers[i]
        idx = len(self.events)
        for key, val in reasoning.items():
            self.reasoning[f"{idx}:{key}"] = val
        self.emit({"type": "final", "skipped": False, "subject": final["subject"], "qid": final["qid"],
                   "qualifiers": qualifiers, "wagers": wagers, "wager_info": wager_info,
                   "replies": [asdict(r) for r in replies], "scores_before": before,
                   "scores_after": list(self.scores)})

    # -- helpers --------------------------------------------------------------
    def ctx(self, r_i: int) -> dict:
        return {"names": self.names, "scores": list(self.scores), "round": self.board.rounds[r_i].name}

    def clue_ctx(self, cell: Cell) -> dict:
        return {"subject": cell.subject, "value": cell.value, "qid": cell.qid,
                "question": self.q.at[cell.qid, "question"]}

    def ask_all(self, ask: Callable) -> tuple[list[Reply], dict]:
        outcomes = list(self.pool.map(ask, self.players))
        replies, reasoning = [], {}
        for i, out in enumerate(outcomes):
            d = out.data or {}
            replies.append(Reply(
                player=i, response=d.get("response", ""), explanation=d.get("explanation", ""),
                confidence=d.get("confidence", 0), buzz=d.get("buzz", False), quip=d.get("quip", ""),
                latency_ms=out.latency_ms, tokens=out.tokens, reasoning_tokens=out.reasoning_tokens,
                error=out.error))
            if out.reasoning:
                reasoning[str(i)] = out.reasoning
        return replies, reasoning

    def judge_all(self, qid: str, replies: list[Reply]) -> None:
        question, answer = self.q.at[qid, "question"], self.q.at[qid, "answer"]
        answer_type = self.q.at[qid, "answer_type"] if "answer_type" in self.q.columns else "exactMatch"
        verdicts = self.pool.map(lambda r: self.judge.grade(question, answer, r.response, answer_type), replies)
        for r, v in zip(replies, verdicts):
            r.verdict = v

    def emit_clue(self, r_i, cell, replies, reasoning, buzz_order, wager, dd=None) -> None:
        idx = len(self.events)
        for key, val in reasoning.items():
            self.reasoning[f"{idx}:{key}"] = val
        self.emit({"type": "clue", "round": r_i, "subject": cell.subject, "value": cell.value,
                   "row": cell.row, "col": cell.col, "qid": cell.qid, "daily_double": cell.daily_double,
                   "wager": wager, "dd": dd, "replies": [asdict(r) for r in replies],
                   "buzz_order": buzz_order, "scores_after": list(self.scores), "control_after": self.control})

    def emit(self, event: dict) -> None:
        self.events.append(event)
        self.on_event(event)
