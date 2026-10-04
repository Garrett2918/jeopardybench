import pytest

import stats


def reply(player, ok, conf, buzz, error=None):
    return {"player": player, "confidence": conf, "buzz": buzz, "error": error, "verdict": {"correct": ok}}


LOG = {
    "players": [{"model": "a"}, {"model": "b"}],
    "events": [
        {"type": "clue", "value": 200, "daily_double": False,
         "replies": [reply(0, True, 90, True), reply(1, False, 80, True)]},
        {"type": "clue", "value": 400, "daily_double": False,
         "replies": [reply(0, False, 30, False), reply(1, True, 60, False)]},
        {"type": "clue", "value": 600, "daily_double": True, "wager": 500, "dd": {"player": 0, "max_wager": 1000},
         "replies": [reply(0, True, 70, True), reply(1, False, 0, False, error="timeout")]},
        {"type": "final", "skipped": False, "qualifiers": [0], "wagers": [350, None],
         "scores_before": [700, -200], "replies": [reply(0, False, 50, True), reply(1, True, 90, False)]},
    ],
    "result": {"scores": [350, -200], "winners": [0]},
    "usage": {"per_player": [{"failures": 0, "tokens_in": 10, "tokens_out": 5},
                             {"failures": 1, "tokens_in": 20, "tokens_out": 5}]},
}


def test_metrics():
    rows = {r["model"]: r for r in stats.compute([LOG])}
    a, b = rows["a"], rows["b"]
    assert a["clues"] == 4 and a["accuracy"] == 0.5
    assert a["buzz_rate"] == 0.5 and a["buzz_accuracy"] == 1.0
    assert b["buzz_rate"] == 0.5 and b["buzz_accuracy"] == 0.0
    assert a["mean_sealed_score"] == 200 - 400 + 600
    assert b["mean_sealed_score"] == -200 + 400 - 600
    assert a["wager_frac"] == pytest.approx((0.5 + 0.5) / 2) and a["wager_win_rate"] == 0.5
    assert a["wins"] == 1 and b["wins"] == 0 and b["failures"] == 1 and b["tokens"] == 25
    # b's errored reply is excluded from calibration: (0.8, F), (0.6, T), (0.9, T)
    assert b["conf_when_right"] == pytest.approx(75) and b["conf_when_wrong"] == pytest.approx(80)


def test_ece():
    assert stats.ece([(0.9, True), (0.9, False)]) == pytest.approx(0.4)
    assert stats.ece([(0.05, False), (0.95, True)]) == pytest.approx(0.05)
    assert stats.ece([]) is None


def test_cli(tmp_path, capsys):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps(LOG))
    stats.main([str(p), "--csv", str(tmp_path / "s.csv")])
    out = capsys.readouterr().out
    assert "buzz_accuracy" in out and (tmp_path / "s.csv").exists()
