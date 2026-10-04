"""Seeded board generation. Spec §5.1."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .data import eligible_subjects
from .log import Cell, Round

ROUND_SPECS = [
    ("Jeopardy", [200, 400, 600, 800, 1000], 1),
    ("Double Jeopardy", [400, 800, 1200, 1600, 2000], 2),
]
COLUMNS = 6


@dataclass
class GameBoard:
    rounds: list[Round]
    final: dict  # {"subject", "qid"}


def build_game_board(df: pd.DataFrame, rng: random.Random, short: bool = False) -> GameBoard:
    specs = ROUND_SPECS[:1] if short else ROUND_SPECS
    subjects = eligible_subjects(df)
    needed = COLUMNS * len(specs) + 1
    if len(subjects) < needed:
        raise ValueError(f"need {needed} eligible subjects, have {len(subjects)}")
    picked = rng.sample(subjects, needed)
    by_subject = {s: sorted(g.index) for s, g in df.groupby("raw_subject")}

    rounds = []
    for r_i, (name, values, n_dd) in enumerate(specs):
        round_subjects = picked[r_i * COLUMNS:(r_i + 1) * COLUMNS]
        dd_cols = rng.sample(range(COLUMNS), n_dd)
        dd_cells = {(rng.randrange(1, len(values)), c) for c in dd_cols}
        cells = []
        for col, subject in enumerate(round_subjects):
            qids = rng.sample(by_subject[subject], len(values))
            for row, (qid, value) in enumerate(zip(qids, values)):
                cells.append(Cell(subject, value, qid, (row, col) in dd_cells, row, col))
        rounds.append(Round(name, values, round_subjects, cells))

    final_subject = picked[-1]
    exact = [q for q in by_subject[final_subject] if df.at[q, "answer_type"] == "exactMatch"]
    if not exact:
        used = {s for r in rounds for s in r.subjects} | {final_subject}
        exact = sorted(df[(~df["raw_subject"].isin(used)) & (df["answer_type"] == "exactMatch")].index)
        final_subject = None
    qid = rng.choice(exact)
    return GameBoard(rounds, {"subject": final_subject or df.at[qid, "raw_subject"], "qid": qid})
