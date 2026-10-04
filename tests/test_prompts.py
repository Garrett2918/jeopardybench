import pytest

from jeopardybench import prompts as P


def test_extract_plain_fenced_and_prose():
    assert P.extract_json('{"a": 1}') == {"a": 1}
    assert P.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert P.extract_json('Sure! Here: {"a": {"b": "}"}} hope that helps') == {"a": {"b": "}"}}
    assert P.extract_json('bad {x} then {"a": 2}') == {"a": 2}


@pytest.mark.parametrize("text", ["", None, "no json here", "[1, 2]"])
def test_extract_rejects(text):
    with pytest.raises(ValueError):
        P.extract_json(text)


def test_parse_clue_clamps_and_cleans():
    r = P.parse_clue('{"response": " What is 4? ", "confidence": "140%", "buzz": "yes", '
                     '"quip": "' + "word " * 30 + '"}')
    assert r["response"] == "What is 4?"
    assert r["confidence"] == 100 and r["buzz"] is True
    assert len(r["quip"].split()) == 15
    assert P.parse_clue('{"response": "x", "confidence": -5, "buzz": false}')["confidence"] == 0


@pytest.mark.parametrize("text", [
    '{"confidence": 50, "buzz": true}',                 # no response
    '{"response": "x", "confidence": "high", "buzz": true}',
    '{"response": "x", "confidence": 50, "buzz": "maybe"}',
    '{"response": "x", "confidence": 50}',              # buzz missing on a regular clue
])
def test_parse_clue_invalid(text):
    with pytest.raises(ValueError):
        P.parse_clue(text)


def test_forced_clue_needs_no_buzz():
    assert P.parse_clue('{"response": "x", "confidence": 50}', forced=True)["buzz"] is True


def test_wager_clamped():
    assert P.parse_wager('{"wager": "$99,999"}', 5, 1000)["wager"] == 1000
    assert P.parse_wager('{"wager": 1}', 5, 1000)["wager"] == 5
    with pytest.raises(ValueError):
        P.parse_wager('{"wager": "all"}', 5, 1000)


def test_judge_and_pick():
    assert P.parse_judge('{"correct": true, "extracted_answer": "4"}')["correct"] is True
    assert P.parse_pick('{"subject": "Chess", "value": "$800"}') == {"subject": "Chess", "value": 800}


def test_builders_include_key_info():
    ctx = {"name": "A", "names": ["A", "B"], "scores": [200, -400], "round": "Jeopardy",
           "subject": "Chess", "value": 800, "question": "Q?", "wager": 300, "max_wager": 1000,
           "remaining": {"Chess": [200, 400]}}
    text = P.clue(ctx)[1]["content"]
    assert "Chess" in text and "$800" in text and "B -$400" in text and "Q?" in text
    assert "LOSE" in P.clue(ctx)[0]["content"]
    assert "$300" in P.forced_answer(ctx, "dd")[1]["content"]
    assert "$1,000" in P.wager(ctx, "dd")[1]["content"]
    assert "Q?" not in P.wager(ctx, "final")[1]["content"]
