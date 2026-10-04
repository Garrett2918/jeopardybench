"""Game log data types and JSON (de)serialisation. Format: spec §7."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOG_VERSION = 1


@dataclass
class Verdict:
    correct: bool
    extracted_answer: str = ""
    reasoning: str = ""
    judge_error: bool = False
    method: str = "judge"  # letter | judge | judge x2 | judge x3 | empty
    votes: list[bool] | None = None


@dataclass
class Reply:
    player: int
    response: str = ""
    explanation: str = ""
    confidence: int = 0
    buzz: bool = False
    quip: str = ""
    latency_ms: int = 0
    tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    reasoning_tokens: int | None = None
    error: str | None = None
    verdict: Verdict | None = None
    counted: bool = False


@dataclass
class Cell:
    subject: str
    value: int
    qid: str
    daily_double: bool = False
    row: int = 0
    col: int = 0


@dataclass
class Round:
    name: str
    values: list[int]
    subjects: list[str]
    cells: list[Cell]


@dataclass
class Player:
    name: str
    model: str
    color: str


@dataclass
class GameLog:
    id: str
    seed: int
    config: dict
    dataset: dict
    players: list[Player]
    rounds: list[Round]
    final: dict | None
    questions: dict[str, dict]
    events: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    version: int = LOG_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> GameLog:
        d = dict(d)
        d["players"] = [Player(**p) for p in d["players"]]
        d["rounds"] = [
            Round(**{**r, "cells": [Cell(**c) for c in r["cells"]]}) for r in d["rounds"]
        ]
        return cls(**d)

    @classmethod
    def load(cls, path: Path) -> GameLog:
        return cls.from_dict(json.loads(Path(path).read_text()))


def reply_to_dict(r: Reply) -> dict:
    return asdict(r)
