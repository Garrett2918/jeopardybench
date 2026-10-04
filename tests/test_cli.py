import json
import os

import pytest

import run
from jeopardybench.log import GameLog
from jeopardybench.players import Outcome


def test_mock_game_writes_valid_log(tmp_path, capsys):
    run.main(["--mock", "--seed", "1", "--out", str(tmp_path)])
    logs = [p for p in tmp_path.glob("*.json") if not p.name.endswith(".reasoning.json")]
    assert len(logs) == 1
    log = GameLog.load(logs[0])
    clues = [e for e in log.events if e["type"] == "clue"]
    assert len(clues) == 60 and log.events[-1]["type"] == "final"
    assert sum(e["daily_double"] for e in clues) == 3
    assert set(log.questions) >= {e["qid"] for e in clues}
    assert log.result["scores"] == log.events[-1]["scores_after"]
    text = logs[0].read_text()
    for var in ("SURPLUS_API_KEY", "KOVAKS_API_KEY", "HF_TOKEN"):
        secret = os.environ.get(var)
        assert not secret or secret not in text  # env var names may appear, values never
    reasoning = json.loads(logs[0].with_name(logs[0].stem + ".reasoning.json").read_text())
    assert reasoning and all(":" in k for k in reasoning)
    assert "Result" in capsys.readouterr().out


def test_short_game(tmp_path):
    run.main(["--mock", "--seed", "2", "--short", "--out", str(tmp_path)])
    log = GameLog.load(next(p for p in tmp_path.glob("*.json") if "reasoning" not in p.name))
    assert len(log.rounds) == 1
    assert len([e for e in log.events if e["type"] == "clue"]) == 30


def test_preflight_failure_exits_before_game(monkeypatch):
    class P:
        def __init__(self, ok):
            self.model, self.ok = "m", ok

        def preflight(self):
            return Outcome({"ok": True} if self.ok else None, error=None if self.ok else "HTTP 404")

    with pytest.raises(SystemExit):
        run.preflight([P(True), P(False)], judge=object())
