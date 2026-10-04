"""Grading. Spec §6 (judge prompt) and §5.6 (judge failure).

- Multiple choice: if the response names exactly one valid option letter, grade by letter
  with no model call. Otherwise (e.g. the option's text instead of its letter) use the judge.
- Exact answers: two independent judge calls; if they disagree, a third breaks the tie.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .log import Verdict
from .players import Player

# "What is F?", "What is E. \mathbb{R}?", "What is A (Logistic Regression)?", "What is D: low EQE…"
LETTER = re.compile(
    r"^\s*(?:(?:what|who|which)\s+(?:is|are)\s+)?(?:option\s+|answer\s+|choice\s+)?\(?([A-Z])\)?"
    r"(?=\s*$|\s*[?.:)]|\s+\(|\s+[—–-]\s|,\s+(?![A-Z]\b))",
    re.IGNORECASE)
CHOICE = re.compile(r"^\s*([A-Z])[.)]\s+(.*\S)\s*$", re.MULTILINE)
PREFIX = re.compile(r"^\s*(?:(?:what|who|which)\s+(?:is|are)\s+)?(?:the\s+)?", re.IGNORECASE)


def options(question: str) -> dict[str, str]:
    """{letter: option text} from the question's 'Answer Choices:' block."""
    _, sep, choices = question.rpartition("Answer Choices:")
    return dict(CHOICE.findall(choices)) if sep else {}


def option_letters(question: str) -> set[str]:
    return set(options(question))


def _norm(text: str) -> str:
    return " ".join(PREFIX.sub("", text).strip().rstrip("?.!").strip().strip("*\"'“”").lower().split())


def letter_of(response: str, valid: set[str] | dict[str, str]) -> str | None:
    """The single option letter a response names, by letter or by the option's exact text."""
    m = LETTER.match(response.strip())
    if not m:
        if isinstance(valid, dict):  # the response may quote an option's text verbatim
            said = _norm(response)
            hits = [k for k, text in valid.items() if _norm(text) == said]
            return hits[0] if len(hits) == 1 else None
        return None
    letter = m.group(1)
    # A lowercase single letter is only accepted when it can't be an English word ("a", "i").
    if letter.islower() and letter in ("a", "i"):
        return None
    letter = letter.upper()
    return letter if letter in valid else None


class Judge:
    def __init__(self, player: Player):
        self.player = player
        self.pool = ThreadPoolExecutor(max_workers=8)  # own pool: the game's pool calls grade()

    @property
    def model(self) -> str:
        return self.player.model

    @property
    def usage(self) -> dict:
        return self.player.usage

    def grade(self, question: str, answer: str, response: str, answer_type: str = "exactMatch") -> Verdict:
        if not response.strip():
            return Verdict(False, "", "Empty response.", method="empty")
        if answer_type == "multipleChoice":
            letter = letter_of(response, options(question))
            if letter:
                ok = letter == answer.strip().upper()
                return Verdict(ok, letter, f"Chose {letter}; the answer is {answer.strip()}.", method="letter")
            return self.ask(question, answer, response)  # no clean letter: let the model read it
        first = list(self.pool.map(lambda _: self.ask(question, answer, response), range(2)))
        votes = [v for v in first if not v.judge_error]
        if len(votes) == 2 and votes[0].correct != votes[1].correct:
            third = self.ask(question, answer, response)
            if not third.judge_error:
                votes.append(third)
        if not votes:
            return first[0]
        n_true = sum(v.correct for v in votes)
        correct = n_true * 2 > len(votes)
        chosen = next(v for v in votes if v.correct == correct)
        return Verdict(correct, chosen.extracted_answer, chosen.reasoning,
                       method=f"judge x{len(votes)}", votes=[v.correct for v in votes])

    def ask(self, question: str, answer: str, response: str) -> Verdict:
        out = None
        for _ in range(2):
            out = self.player.call(P.judge(question, answer, response), P.parse_judge,
                                   '{"extracted_answer": "...", "reasoning": "...", "correct": true}')
            if out.data:
                return Verdict(**out.data, method="judge")
        return Verdict(False, "", f"Judge failed: {out.error}", judge_error=True, method="judge")
