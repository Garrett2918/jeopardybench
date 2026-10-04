import random

import pandas as pd

from jeopardybench.board import GameBoard
from jeopardybench.game import Game
from jeopardybench.log import Cell, Round, Verdict
from jeopardybench.players import Outcome


class Scripted:
    """Answers from a script: {qid: (response, confidence, buzz, latency)}; default no buzz."""

    def __init__(self, name, script=None, wager=0, pick=None):
        self.name, self.model, self.color = name, "m-" + name, "#000"
        self.script, self.desired_wager, self.pick_reply = script or {}, wager, pick
        self.calls, self.wager_bounds = [], []

    def _ans(self, ctx, forced):
        self.calls.append(("forced" if forced else "clue", ctx["qid"]))
        resp, conf, buzz, lat = self.script.get(ctx["qid"], ("What is nothing?", 0, False, 1000))
        return Outcome({"response": resp, "explanation": "", "confidence": conf,
                        "buzz": True if forced else buzz, "quip": ""}, latency_ms=lat)

    def answer_clue(self, ctx):
        return self._ans(ctx, False)

    def answer_forced(self, ctx, kind):
        return self._ans(ctx, True)

    def pick(self, ctx):
        if self.pick_reply:
            return Outcome(self.pick_reply)
        s = sorted(ctx["remaining"])[0]
        return Outcome({"subject": s, "value": ctx["remaining"][s][0]})

    def wager(self, ctx, kind, low, high):
        self.wager_bounds.append((kind, low, high))
        return Outcome({"wager": max(low, min(high, self.desired_wager)), "quip": ""})


class Judge:
    def grade(self, q, a, response, answer_type="exactMatch"):
        return Verdict(response == "What is RIGHT?")


R, W = "What is RIGHT?", "What is WRONG?"


def make_game(players, cells_per_round, final_qid="fq", seed=0):
    qids = [c.qid for cells in cells_per_round for c in cells] + [final_qid]
    df = pd.DataFrame({"question": [f"Q {q}" for q in qids], "answer": ["RIGHT"] * len(qids)}, index=qids)
    rounds = [Round(f"R{i}", [200, 400, 600, 800, 1000], sorted({c.subject for c in cells}), cells)
              for i, cells in enumerate(cells_per_round)]
    return Game(GameBoard(rounds, {"subject": "F", "qid": final_qid}), df, players, Judge(), random.Random(seed))


def clue_events(g):
    return [e for e in g.events if e["type"] == "clue"]


def test_wrong_then_right_buzz_chain():
    players = [Scripted("A", {"q1": (W, 90, True, 5000)}), Scripted("B", {"q1": (R, 80, True, 100)}),
               Scripted("C", {"q1": (R, 70, False, 100)})]
    g = make_game(players, [[Cell("S", 400, "q1")]])
    g.play_round(0)
    e = clue_events(g)[0]
    assert e["buzz_order"] == [0, 1]
    assert g.scores == [-400, 400, 0] and g.control == 1
    counted = [r["counted"] for r in e["replies"]]
    assert counted == [True, True, False]
    assert e["replies"][2]["verdict"]["correct"] is True  # judged even though it didn't count


def test_nobody_buzzes_keeps_control():
    players = [Scripted("A"), Scripted("B")]
    g = make_game(players, [[Cell("S", 200, "q1")]], seed=3)
    g.play_round(0)
    start = g.events[0]["control"]
    assert g.scores == [0, 0] and g.control == start


def test_confidence_tie_broken_by_latency():
    players = [Scripted("A", {"q1": (R, 80, True, 9000)}), Scripted("B", {"q1": (R, 80, True, 10)})]
    g = make_game(players, [[Cell("S", 200, "q1")]])
    g.play_round(0)
    assert g.scores == [0, 200]


def test_daily_double_win_loss_and_bounds():
    a = Scripted("A", {"q1": (R, 99, True, 1), "dd1": (W, 50, True, 1)}, wager=99999)
    b = Scripted("B", {"dd1": (R, 99, True, 1)})
    g = make_game([a, b], [[Cell("S", 200, "q1"), Cell("S", 400, "dd1", daily_double=True, row=1)]])
    g.play_round(0)
    # A won q1 (+200) and control, then DD: max wager = max(200, 1000) = 1000, wrong → -1000
    assert a.wager_bounds == [("dd", 5, 1000)]
    dd = clue_events(g)[1]
    assert dd["wager"] == 1000 and dd["buzz_order"] == [0]
    assert g.scores == [-800, 0]  # B's correct answer on the DD doesn't count
    assert dd["replies"][1]["counted"] is False and dd["replies"][1]["verdict"]["correct"] is True
    assert ("forced", "dd1") in a.calls and ("clue", "dd1") in b.calls
    assert g.control == 0


def test_second_round_starts_with_lowest_scorer():
    players = [Scripted("A", {"q1": (R, 90, True, 1)}), Scripted("B"), Scripted("C", {"q1": (W, 80, True, 1)})]
    g = make_game(players, [[Cell("S", 200, "q1")], [Cell("T", 400, "q2")]])
    g.play_round(0)
    g.play_round(1)
    assert g.scores == [200, 0, 0]  # A buzzed first and was right; C never reached
    starts = [e for e in g.events if e["type"] == "round_start"]
    assert starts[1]["control"] in (1, 2)


def test_final_sits_out_nonpositive_and_co_champions():
    a = Scripted("A", {"q1": (R, 90, True, 1), "fq": (R, 50, True, 1)}, wager=300)
    b = Scripted("B", {"q1": (W, 95, True, 1), "fq": (R, 99, True, 1)}, wager=100)
    c = Scripted("C", {"q2": (R, 90, True, 1), "fq": (W, 50, True, 1)}, wager=200)
    g = make_game([a, b, c], [[Cell("S", 200, "q1"), Cell("S", 400, "q2")]])
    result = g.play()
    final = g.events[-1]
    # before final: A +200, B -200, C +400. B sits out.
    assert final["scores_before"] == [200, -200, 400]
    assert final["qualifiers"] == [0, 2] and final["wagers"] == [200, None, 200]  # A capped at its score
    assert ("clue", "fq") in b.calls and final["replies"][1]["counted"] is False
    assert result == {"scores": [400, -200, 200], "winners": [0]}


def test_tie_gives_co_champions_and_skipped_final():
    g = make_game([Scripted("A"), Scripted("B")], [[Cell("S", 200, "q1")]])
    result = g.play()
    assert g.events[-1]["skipped"] is True
    assert result["winners"] == [0, 1]


def test_invalid_pick_falls_back():
    a = Scripted("A", pick={"subject": "Nope", "value": 9})
    g = make_game([a], [[Cell("S", 400, "q2"), Cell("S", 200, "q1", row=0)]])
    g.play_round(0)
    picks = [e for e in g.events if e["type"] == "pick"]
    assert picks[0]["fallback"] and picks[0]["value"] == 200 and picks[0]["error"] == "invalid pick"


def test_failed_reply_is_no_buzz():
    class Broken(Scripted):
        def answer_clue(self, ctx):
            return Outcome(None, error="timeout")
    g = make_game([Broken("A"), Scripted("B")], [[Cell("S", 200, "q1")]])
    g.play_round(0)
    r = clue_events(g)[0]["replies"][0]
    assert r["buzz"] is False and r["error"] == "timeout" and r["verdict"]["correct"] is False
