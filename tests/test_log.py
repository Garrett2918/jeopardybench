from jeopardybench.log import Cell, GameLog, Player, Round


def test_round_trip(tmp_path):
    log = GameLog(
        id="g1", seed=1, config={"a": 1}, dataset={"file": "x"},
        players=[Player("A", "m-a", "#fff")],
        rounds=[Round("Jeopardy", [200], ["S"], [Cell("S", 200, "q1", True, 1, 0)])],
        final={"subject": "S2", "qid": "q2"},
        questions={"q1": {"question": "?", "answer": "1"}},
        events=[{"type": "pick"}],
    )
    p = tmp_path / "g.json"
    log.save(p)
    back = GameLog.load(p)
    assert back == log
    assert back.rounds[0].cells[0].daily_double is True
