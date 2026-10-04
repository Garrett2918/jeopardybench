"""Prompt templates and reply validation. Spec §6.

Every builder returns OpenAI chat messages; every validator takes the raw model text
and returns a cleaned dict or raises ValueError.
"""
from __future__ import annotations

import json
import re

JUDGE_PROMPT_VERSION = 1

SYSTEM = """You are {name}, a contestant on JeopardyBench, a Jeopardy-style quiz show between AI models.
Clues are taken from Humanity's Last Exam, an extremely hard expert-level exam.

Scoring on a regular clue:
- If you buzz and your response is correct, you EARN the clue's value.
- If you buzz and your response is WRONG, you LOSE the clue's value.
- If you do not buzz, your score does not change.
Players who buzz are ranked by their stated confidence (0-100), highest first; the first correct
buzzer takes the points. State your confidence honestly: it is the probability you are right.

Phrase every response as a question ("What is ...?"). Always reply with a single JSON object
and nothing else."""

CLUE_SCHEMA = (
    '{"response": "What is ...?", "explanation": "1-3 sentences on why", '
    '"confidence": 0-100, "buzz": true or false, "quip": "optional table talk, max 15 words"}'
)
FORCED_SCHEMA = (
    '{"response": "What is ...?", "explanation": "1-3 sentences on why", '
    '"confidence": 0-100, "quip": "optional table talk, max 15 words"}'
)


def _scores(names: list[str], scores: list[int]) -> str:
    return ", ".join(f"{n} {'-' if s < 0 else ''}${abs(s):,}" for n, s in zip(names, scores))


def system(name: str) -> dict:
    return {"role": "system", "content": SYSTEM.format(name=name)}


def clue(ctx: dict) -> list[dict]:
    return [system(ctx["name"]), {"role": "user", "content": (
        f"Round: {ctx['round']}. Scores: {_scores(ctx['names'], ctx['scores'])}.\n"
        f"Category: {ctx['subject']}. Value: ${ctx['value']:,}.\n\n"
        f"Clue:\n{ctx['question']}\n\n"
        f"Decide whether to buzz. Reply with JSON only: {CLUE_SCHEMA}"
    )}]


def forced_answer(ctx: dict, kind: str) -> list[dict]:
    """Daily Double or Final Jeopardy answer: the player must respond."""
    stake = f"You wagered ${ctx['wager']:,}." if ctx.get("wager") is not None else ""
    label = "DAILY DOUBLE" if kind == "dd" else "FINAL JEOPARDY"
    return [system(ctx["name"]), {"role": "user", "content": (
        f"{label}. Round: {ctx['round']}. Scores: {_scores(ctx['names'], ctx['scores'])}.\n"
        f"Category: {ctx['subject']}. {stake} You must respond (no passing); "
        f"a correct response wins your wager, a wrong one loses it.\n\n"
        f"Clue:\n{ctx['question']}\n\nReply with JSON only: {FORCED_SCHEMA}"
    )}]


def pick(ctx: dict) -> list[dict]:
    board = "\n".join(f"- {s}: {', '.join(f'${v}' for v in vals)}" for s, vals in ctx["remaining"].items())
    return [system(ctx["name"]), {"role": "user", "content": (
        f"Round: {ctx['round']}. Scores: {_scores(ctx['names'], ctx['scores'])}.\n"
        f"You control the board. Remaining clues:\n{board}\n\n"
        'Pick the next clue. Reply with JSON only: {"subject": "<category name>", "value": <dollar amount>}'
    )}]


def wager(ctx: dict, kind: str) -> list[dict]:
    if kind == "dd":
        rule = (f"DAILY DOUBLE! You may wager between $5 and ${ctx['max_wager']:,}. "
                "You will then have to answer the clue; you only know its category.")
    else:
        rule = (f"FINAL JEOPARDY. You may wager between $0 and ${ctx['max_wager']:,}. "
                "Everyone with a positive score answers the same clue; highest total wins.")
    return [system(ctx["name"]), {"role": "user", "content": (
        f"Round: {ctx['round']}. Scores: {_scores(ctx['names'], ctx['scores'])}.\n"
        f"Category: {ctx['subject']}.\n{rule}\n\n"
        'Reply with JSON only: {"wager": <integer>, "quip": "optional, max 15 words"}'
    )}]


JUDGE = """Judge whether the following [response] to [question] is correct based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {answer}

Extract the final answer from the response, ignoring Jeopardy phrasing such as "What is". The response is correct only if its final answer matches [correct_answer]; allow small numerical tolerance and mathematically or semantically equivalent forms, nothing else. For multiple choice, the chosen option must match. An empty response is incorrect.
Reply with JSON only: {{"extracted_answer": "<string>", "reasoning": "<1-2 sentences>", "correct": true or false}}"""


def judge(question: str, answer: str, response: str) -> list[dict]:
    return [{"role": "user", "content": JUDGE.format(question=question, answer=answer, response=response or "(no response)")}]


def preflight() -> list[dict]:
    return [{"role": "user", "content": 'Reply with JSON only: {"ok": true, "answer": "<the capital of France>"}'}]


# ---- parsing -------------------------------------------------------------

def extract_json(text: str | None) -> dict:
    if not text:
        raise ValueError("empty reply")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
        start = text.find("{")
        while start != -1 and obj is None:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("{", start + 1)
    if not isinstance(obj, dict):
        raise ValueError("no JSON object in reply")
    return obj


def _int(v, name) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a number")
    try:
        return int(round(float(str(v).replace("$", "").replace(",", "").replace("%", ""))))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None


def _bool(v, name) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
        return v.strip().lower() in ("true", "yes")
    raise ValueError(f"{name} must be true/false")


def _quip(v) -> str:
    return " ".join(str(v or "").split()[:15])


def parse_clue(text: str, forced: bool = False) -> dict:
    o = extract_json(text)
    if not isinstance(o.get("response"), str):
        raise ValueError("missing response")
    out = {
        "response": o["response"].strip()[:2000],
        "explanation": str(o.get("explanation") or "").strip()[:600],
        "confidence": max(0, min(100, _int(o.get("confidence"), "confidence"))),
        "quip": _quip(o.get("quip")),
    }
    out["buzz"] = True if forced else _bool(o.get("buzz"), "buzz")
    return out


def parse_pick(text: str) -> dict:
    o = extract_json(text)
    return {"subject": str(o.get("subject", "")).strip(), "value": _int(o.get("value"), "value")}


def parse_wager(text: str, low: int, high: int) -> dict:
    o = extract_json(text)
    return {"wager": max(low, min(high, _int(o.get("wager"), "wager"))), "quip": _quip(o.get("quip"))}


def parse_judge(text: str) -> dict:
    o = extract_json(text)
    return {
        "correct": _bool(o.get("correct"), "correct"),
        "extracted_answer": str(o.get("extracted_answer") or "")[:500],
        "reasoning": str(o.get("reasoning") or "")[:1000],
    }


def parse_preflight(text: str) -> dict:
    o = extract_json(text)
    if o.get("ok") is not True:
        raise ValueError("preflight reply missing ok: true")
    return o
