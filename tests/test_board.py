import random

import pytest

from jeopardybench.board import build_game_board


def board(questions, seed=1, short=False):
    return build_game_board(questions, random.Random(seed), short)


def test_deterministic(questions):
    assert board(questions, 7) == board(questions, 7)
    assert board(questions, 7) != board(questions, 8)


def test_shape_and_subjects(questions):
    b = board(questions)
    assert [len(r.cells) for r in b.rounds] == [30, 30]
    all_subjects = [s for r in b.rounds for s in r.subjects] + [b.final["subject"]]
    assert len(set(all_subjects)) == 13
    for r in b.rounds:
        for c in r.cells:
            assert questions.at[c.qid, "raw_subject"] == c.subject
            assert c.value == r.values[c.row]
    qids = [c.qid for r in b.rounds for c in r.cells] + [b.final["qid"]]
    assert len(set(qids)) == len(qids)


@pytest.mark.parametrize("seed", range(20))
def test_daily_doubles(questions, seed):
    b = board(questions, seed)
    for r, n in zip(b.rounds, [1, 2]):
        dds = [c for c in r.cells if c.daily_double]
        assert len(dds) == n
        assert all(c.row >= 1 for c in dds)
        assert len({c.col for c in dds}) == n


def test_final_is_exact_match(questions):
    for seed in range(20):
        b = board(questions, seed)
        assert questions.at[b.final["qid"], "answer_type"] == "exactMatch"


def test_short(questions):
    b = board(questions, short=True)
    assert len(b.rounds) == 1


def test_not_enough_subjects(questions):
    with pytest.raises(ValueError):
        build_game_board(questions[questions.raw_subject.isin(["Chess", "Law"])], random.Random(1))
