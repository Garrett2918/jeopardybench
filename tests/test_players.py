import pytest

from jeopardybench.judge import Judge
from jeopardybench.players import CallError, Completion, Player, extract_reasoning

GOOD = '{"response": "What is 4?", "explanation": "x", "confidence": 80, "buzz": true}'
CTX = {"name": "A", "names": ["A"], "scores": [0], "round": "Jeopardy", "subject": "S",
       "value": 200, "question": "Q"}


class FakeTransport:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def complete(self, model, messages, params):
        self.calls.append(messages)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, Completion) else Completion(r, tokens_in=10, tokens_out=5, latency_ms=100)


def test_success():
    p = Player("A", "m", "#000", FakeTransport(GOOD))
    out = p.answer_clue(CTX)
    assert out.data["confidence"] == 80 and out.error is None
    assert p.usage == {"calls": 1, "tokens_in": 10, "tokens_out": 5, "failures": 0}


def test_retry_after_bad_json_then_success():
    t = FakeTransport("I think it's 4", GOOD)
    out = Player("A", "m", "#000", t).answer_clue(CTX)
    assert out.data and out.tokens == {"in": 20, "out": 10} and out.latency_ms == 200
    assert "JSON only" in t.calls[1][-1]["content"]


def test_two_bad_replies_fail():
    p = Player("A", "m", "#000", FakeTransport("nope", "still nope"))
    out = p.answer_clue(CTX)
    assert out.data is None and out.error.startswith("bad reply")
    assert p.usage["failures"] == 1


def test_call_error_no_retry():
    t = FakeTransport(CallError("timeout"), GOOD)
    out = Player("A", "m", "#000", t).answer_clue(CTX)
    assert out.data is None and out.error == "timeout" and len(t.calls) == 1


def test_reasoning_kind():
    c = Completion(GOOD, reasoning="thinking...")
    assert Player("A", "m", "#0", FakeTransport(c), native_reasoning=True).answer_clue(CTX).reasoning == \
        {"kind": "native", "text": "thinking..."}
    c = Completion(GOOD, reasoning="summary")
    assert Player("A", "m", "#0", FakeTransport(c)).answer_clue(CTX).reasoning["kind"] == "summary"


def test_extract_reasoning_variants():
    assert extract_reasoning({"reasoning_content": "a"}) == "a"
    assert extract_reasoning({"reasoning": "b"}) == "b"
    assert extract_reasoning({"reasoning_details": [{"text": "c"}, {"summary": "d"}]}) == "c\nd"
    assert extract_reasoning({"reasoning_content": "  ", "content": "x"}) is None


def test_judge():
    j = Judge(Player("J", "jm", "#0", FakeTransport(*['{"correct": true, "extracted_answer": "4"}'] * 2)))
    assert j.grade("Q", "4", "What is 4?").correct
    assert not j.grade("Q", "4", "  ").correct  # empty: no API call needed


def test_judge_failure_marks_error():
    j = Judge(Player("J", "jm", "#0", FakeTransport(*[CallError("x")] * 4)))
    v = j.grade("Q", "4", "What is 4?")
    assert v.correct is False and v.judge_error


def test_out_of_tokens_is_not_retried():
    t = FakeTransport(Completion(None, reasoning="x" * 1000, finish_reason="length"), GOOD)
    p = Player("A", "m", "#000", t, {"max_tokens": 32000})
    out = p.answer_clue(CTX)
    assert out.data is None and "ran out of tokens" in out.error and len(t.calls) == 1
    assert out.reasoning["text"]  # the thinking is still kept for the viewer


MCQ = "Pick one.\n\nAnswer Choices:\nA. one\nB. two\nC. three\nD. four"


@pytest.mark.parametrize("response,letter", [
    ("What is B?", "B"), ("What is C. three?", "C"), ("What is D: four", "D"), ("What is A (one)?", "A"),
    ("What is option b?", "B"), ("B", "B"), ("What is E?", None),        # E isn't an option
    ("What is B or C?", None), ("What is a guess?", None), ("What is B, two?", "B"), ("What is A, B?", None),
    ("What is two?", "B"), ("What is the three?", "C"), ("What is twelve?", None),  # option text, verbatim only
])
def test_letter_extraction(response, letter):
    from jeopardybench.judge import letter_of, options
    assert letter_of(response, options(MCQ)) == letter


def test_multiple_choice_graded_without_judge_call():
    t = FakeTransport()
    j = Judge(Player("J", "jm", "#0", t))
    assert j.grade(MCQ, "B", "What is B?", "multipleChoice").correct
    v = j.grade(MCQ, "B", "What is C. three?", "multipleChoice")
    assert not v.correct and v.method == "letter" and t.calls == []


def test_multiple_choice_without_letter_uses_judge():
    t = FakeTransport('{"correct": true, "extracted_answer": "B"}')
    v = Judge(Player("J", "jm", "#0", t)).grade(MCQ, "B", "What is the second one?", "multipleChoice")
    assert v.correct and v.method == "judge" and len(t.calls) == 1


YES, NO = '{"correct": true, "extracted_answer": "x"}', '{"correct": false, "extracted_answer": "x"}'


def test_exact_answer_two_agreeing_votes():
    t = FakeTransport(YES, YES)
    v = Judge(Player("J", "jm", "#0", t)).grade("Q", "4", "What is 4?")
    assert v.correct and v.votes == [True, True] and v.method == "judge x2" and len(t.calls) == 2


def test_exact_answer_disagreement_goes_to_third_vote():
    t = FakeTransport(YES, NO, NO)
    v = Judge(Player("J", "jm", "#0", t)).grade("Q", "4", "What is 4?")
    assert v.correct is False and sorted(v.votes) == [False, False, True] and len(t.calls) == 3
