# JeopardyBench — Design Spec

Date: 2026-10-04
Status: Draft for review

## 1. Purpose

A Jeopardy-style game in which AI models compete on questions from **Humanity's Last Exam (HLE)**.

- **Primary goal:** an entertaining show. Each game is played once against the API and then watched as an animated replay.
- **Secondary goal:** a benchmark. The logs capture every model's answer, confidence, buzz decision and wager, so statistics on knowing when to pass can be computed afterwards.

**Success criteria**
1. One command (`python run.py --seed 42`) plays a full game and writes `games/<id>.json`.
2. Opening `viewer.html?game=games/<id>.json` shows the game as a watchable show.
3. A game is reproducible from its seed and config: same board, same clue order rules.
4. `stats.py` can aggregate many logs into per-model benchmark metrics.

**Non-goals for v1:** image questions, live in-browser games, human players, speed-based buzzing, difficulty-based clue values.

## 2. Decisions

| Topic | Decision |
|---|---|
| Model access | Any OpenAI-compatible endpoint; the default is Surplus Intelligence (`base_url` plus an API key from an env var). |
| Viewing | Run, then replay. A static HTML viewer reads the JSON log. |
| Buzzing | Confidence buzzer: every model answers privately; buzzers are ranked by confidence, ties broken by lower latency. |
| Clue values | Random within a subject. |
| Questions | Text-only HLE questions; subjects with at least 5 text questions (49 subjects, 1,947 questions as of the 2026-10-04 snapshot). |
| Contestants | 3 by default; any number of at least 2 is configurable. |
| Grading | An LLM judge (config `judge`) grades every response; the prompt is adapted from HLE's official grader. |
| Stack | Python 3.11+ (`openai`, `pandas`, `pyarrow`, `pyyaml`) plus one HTML file (KaTeX from cdnjs). |

## 3. Data

- Source: `https://huggingface.co/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet` (gated; requires `HF_TOKEN` and accepted dataset terms).
- Downloaded once and cached at `data/hle.parquet`; `data/` is git-ignored. The token comes from `HF_TOKEN` (env or the project `.env`, see §11).
- Columns used: `id`, `question`, `image`, `answer`, `answer_type` (`exactMatch` | `multipleChoice`), `rationale`, `raw_subject`, `category`.
- Filter: rows with an empty `image`. Eligible subjects: `raw_subject` with at least 5 remaining questions.
- Multiple-choice options are already in the `question` text; no special handling.
- Test fixture: `tests/fixtures/mini_hle.parquet`, about 60 synthetic rows across 8 subjects, used by the tests and `--mock`.

## 4. Architecture

```
jeopardybench/
  config.example.yaml
  jeopardybench/
    data.py      load_questions(path) -> DataFrame; eligible_subjects(df) -> list[str]
    board.py     build_game_board(df, seed) -> GameBoard (2 rounds + final)
    players.py   Player: chat(messages) -> ParsedReply | Failure (OpenAI-compatible client)
    mock.py      MockPlayer / MockJudge with seeded random behaviour
    prompts.py   all prompt templates (system, clue, pick, wager, final, judge)
    judge.py     Judge.grade(question, answer, response) -> Verdict
    game.py      Game(board, players, judge).play() -> GameLog (state machine)
    log.py       GameLog dataclasses + JSON (de)serialisation
  run.py         CLI
  stats.py       aggregate logs -> table (stdout + CSV)
  viewer.html    replay viewer
  tests/
  games/         output logs (git-ignored except examples)
```

Dependencies only point downward: `game` uses `board`, `players`, `judge` and `log`; `players` and `judge` use `prompts`. `game` takes injected player and judge objects, so mocks need no network.

### Config (`config.yaml`)

```yaml
api:
  base_url: https://api.surplusintelligence.ai/v1
  api_key_env: SURPLUS_API_KEY
timeout_s: 600
contestants:                             # initial lineup: cheap "flash" tier
  - name: DeepSeek
    model: deepseek-v4.1-flash
    color: "#4D6BFE"
    params: { max_tokens: 16000 }        # passed through verbatim (e.g. reasoning_effort)
  - name: GLM
    model: glm-5.3-flash
    color: "#7C5CFF"
    params: { max_tokens: 16000 }
  - name: MiMo
    model: mimo-v2.6-flash
    color: "#FF6900"
    params: { max_tokens: 32000 }
judge:
  model: gpt-6-astra                     # not a contestant
  params: { max_tokens: 4000 }
native_reasoning: [deepseek-v4.1-flash, glm-5.3-flash, mimo-v2.6-flash]
```
Notes from probes on 2026-10-04 via Surplus:
- mimo-v2.6-flash returns full native reasoning (~40k chars, 13.6k tokens, 201s on a median HLE clue). Earlier candidates: gemini-3.8-flash (no reasoning) and grok-4.7 (summary only; none when `reasoning_effort` is set).
- Judge cost: ~290–780 input and 55–115 output tokens per grading call, so ~100k in / 15k out per full game (~186 calls), roughly 1–2% of a game's tokens. A stronger judge is cheap, so gpt-6-astra was chosen; it graded 5/5 test cases correctly (incl. an equivalent reordered formula) at ~40–50 output tokens per call.
- Pacing: a clue waits for the slowest contestant, and hard clues take minutes, so a full game may run 2–4 hours. Hence `timeout_s: 600` and progress output in `run.py`.

## 5. Game rules

### 5.1 Board
- **Round 1 (Jeopardy):** 6 subjects × 5 clues, values $200/$400/$600/$800/$1000, 1 Daily Double.
- **Round 2 (Double Jeopardy):** 6 different subjects, $400–$2000, 2 Daily Doubles, placed in different columns.
- **Final Jeopardy:** a 13th subject not used in either round; one `exactMatch` question. If no unused subject has one, use any unused `exactMatch` question.
- Subjects are sampled uniformly without replacement from the eligible subjects; each subject's 5 questions are sampled without replacement and shuffled into rows.
- Daily Doubles are placed at random in rows 2–5 (never row 1).
- Every random choice in board building and the game engine goes through one `random.Random(seed)`. The same seed and config give the same board. Model outputs are not deterministic, so a full game cannot be repeated exactly.
- `--short`: Round 1 plus Final only.

### 5.2 Control and clue selection
- Round 1: a random player has control. Round 2: the lowest scorer (ties broken at random).
- The controlling player gets a **pick** call (remaining board and scores) and returns `{"subject": str, "value": int}`. Invalid or failed reply → the cheapest remaining clue in a random column.
- A correct answer passes control to that player; otherwise control stays put.

### 5.3 Regular clue
1. All players get the **clue** call in parallel: subject, value, round, all scores and the question.
2. Reply: `{"response": str, "explanation": str 1–3 sentences, "confidence": int 0–100, "buzz": bool, "quip": str ≤15 words (optional)}`. Confidence is clamped to 0–100; a quip is cut to 15 words; an explanation is cut to 600 characters.
3. Buzzers are sorted by confidence (descending), then latency (ascending).
4. Going down that order: the judge grades the response; correct → +value, take control, stop; wrong → −value, continue.
5. All non-buzzer responses are also judged (in parallel, after the clue resolves) and logged with `counted: false`.

### 5.4 Daily Double
- Only the controlling player plays. The **wager** call (subject, round, scores) returns `{"wager": int}`, clamped to `[5, max(score, round_max_value)]`.
- The **answer** call uses the clue prompt with the note that it must answer (no `buzz` field). Correct → +wager; wrong → −wager. Control stays with the player either way.
- Other players are still asked the clue (in parallel with the answer call) and judged with `counted: false`, so the benchmark data covers every clue.

### 5.5 Final Jeopardy
- Players with a score above 0 take part. If nobody qualifies, Final is skipped and the highest score wins.
- **Final wager** call (subject, every player's score) → clamped to `[0, score]`. Then the **final answer** call (must answer).
- Non-qualifying players are still asked the question and judged with `counted: false`.
- Reveal order (viewer): lowest pre-Final score first.
- Winner: highest final score; ties → co-champions.

### 5.6 Failures
- **Pre-flight:** before a game, `run.py` sends each contestant and the judge one trivial JSON-format call. If any reply is an error or doesn't parse, the run stops before any game call is made. This catches retired models: on 2026-10-04, `claude-sonnet-4.6` on Surplus returned the text "…no longer available" as a successful reply instead of an error.
- **Out of tokens:** if the model hits `max_tokens` (`finish_reason: length`) before giving a valid answer, it's a fallback (no buzz) with no retry, because the same budget would just repeat. The thinking budget is part of the rules; it is 32k tokens for every contestant. (On 2026-10-04, deepseek-v4.1-flash spent all 32k thinking on one $400 Chemistry clue.)
- A reply that isn't valid JSON or fails the schema → one retry with an appended "Reply with JSON only matching: …" message. A second failure → fallback.
- A timeout or API error → fallback with no retry, except one retry after 5s on HTTP 429/5xx.
- Fallbacks: clue → no buzz, empty response; wager → minimum; final/DD answer → empty response (graded wrong); pick → rule in 5.2; judge → retry once, then `correct: false, judge_error: true`.
- Every failure is recorded on the event (`error` field) and shown in the viewer as a small ⚠.

## 6. Prompts (summary; exact text in `prompts.py`)

- **System (contestants):** game description; clues come from HLE; scoring rules (correct buzz +value, wrong buzz −value, no buzz 0); buzz order by honest confidence; phrase responses as a question; JSON-only reply format.
- **Clue / DD / Final answer / Pick / Wager:** state plus the required JSON schema. No question content in pick or wager calls.
- **Grading (added 2026-10-04):** multiple-choice answers that name exactly one valid option, by letter or by the option's verbatim text, are graded by letter with no model call (51/59 such answers in the first two live games, 100% agreement with the judge). Other multiple-choice answers get one judge call. Exact answers get two independent judge calls, and a third breaks a disagreement; the log records `method` and `votes`. Measured cost: about +19% judge tokens, +2.6% of a game's tokens.
- **Judge:** adapted from the HLE grader: given `[question]`, `[response]`, `[correct_answer]`, extract the final answer (ignoring "What is…"), judge it correct only if it matches the correct answer, allowing small numerical tolerance and equivalent forms; reply JSON `{"extracted_answer", "reasoning", "correct": bool}`.
- The JSON asks for a short `explanation` (a post-hoc justification, available for every model). Native reasoning is never requested in the JSON; reasoning models think natively.

### 6.1 Native reasoning capture
- After each call, `players.py` reads `message.reasoning_content` (fallback: text in `message.reasoning`, then the text parts of `message.reasoning_details`). If present it is stored with `kind`: `"native"` if the model is in the config's `native_reasoning` list, otherwise `"summary"` (length is not a reliable signal: Grok's summaries can run long). `reasoning_tokens` from usage is stored when present.
- Probe on 2026-10-04 via Surplus: full reasoning from deepseek-r1, kimi-k2-thinking, qwen3.5-plus and glm-5.2; short summaries from claude-opus-4.8 and grok-4.5; none from gemini-3.1-pro or the gpt-5.x models.
- Reasoning text goes to a sidecar file `games/<id>.reasoning.json` (`{"<event index>:<player>": {"kind", "text"}}`) so the main log stays small. Judge reasoning stays in the main log (it's short).

## 7. Game log format (`games/<id>.json`)

```jsonc
{
  "version": 1,
  "id": "2026-10-04T05-12-00_seed42",
  "seed": 42,
  "config": { /* contestants, judge, params — no API key */ },
  "dataset": { "file": "hle.parquet", "sha256": "…" },
  "players": [{ "name": "Claude", "model": "…", "color": "#D97757" }],
  "rounds": [
    { "name": "Jeopardy", "values": [200,400,600,800,1000],
      "subjects": ["Chess", …],
      "cells": [{ "subject": "Chess", "value": 800, "qid": "…", "daily_double": false }] }
  ],
  "final": { "subject": "…", "qid": "…" },
  "questions": { "<qid>": { "question": "…", "answer": "…", "answer_type": "…",
                            "rationale": "…", "category": "…" } },
  "events": [
    { "type": "pick", "round": 0, "player": 0, "subject": "Chess", "value": 800,
      "fallback": false },
    { "type": "clue", "round": 0, "subject": "Chess", "value": 800, "qid": "…",
      "daily_double": false, "wager": null,
      "replies": [{ "player": 0, "response": "What is …?", "confidence": 91,
                    "buzz": true, "quip": "…", "latency_ms": 41230,
                    "tokens": { "in": 812, "out": 5320 }, "error": null,
                    "verdict": { "correct": true, "extracted_answer": "…",
                                 "reasoning": "…" },
                    "counted": true }],
      "buzz_order": [0, 2],
      "scores_after": [800, 0, 0], "control_after": 0 },
    { "type": "final", "wagers": [3000, 1200, null], "replies": [ … ],
      "scores_after": [ … ] }
  ],
  "result": { "scores": [ … ], "winners": [0] },
  "usage": { "per_player": [{ "calls": 0, "tokens_in": 0, "tokens_out": 0 }],
             "judge": { … } }
}
```

The log contains everything the viewer needs; the viewer never reads the dataset.

## 8. Replay viewer (`viewer.html`)

- **Loading:** `?game=<url>`, drag and drop, or a file picker. One HTML file; KaTeX (CSS + JS + auto-render) from cdnjs; everything else inline.
- **Layout:** header (round name, controls) / 6×5 board / player podiums (name, model, colour, score, quip bubble). A clue card replaces the board while a clue is open; the question scrolls inside it and math renders with KaTeX.
- **Timeline:** `buildTimeline(log) -> Step[]` is a pure function. Step kinds: `roundIntro`, `pick`, `openClue`, `read`, `buzzReveal`, `response`, `verdict`, `reveal`, `closeClue`, `dailyDoubleSplash`, `wager`, `finalIntro`, `finalReveal`, `endScreen`. The player keeps an index into the steps; each step has a base duration scaled by the speed setting.
- **Clue sequence:** pick caption → zoom in → read bar (duration = clamp(chars / 60 s, 4, 15) at 1×) → buzzers light up in confidence order with % badges, passers dimmed → each judged response in turn with ✓/✗ and an animated score → reveal: correct answer, collapsible "Why" (rationale), "Everyone's answers" strip (all replies with ✓/✗ and their `explanation`, uncounted ones marked) → back to the board.
- **Thinking panel:** each reply in the strip has a "🧠 Thinking" toggle when reasoning exists, labelled "native" or "summary". The sidecar `<id>.reasoning.json` (same URL with `.reasoning.json` in place of `.json`, or a second dropped file) loads on first open; if it's missing, the toggle is hidden. Reasoning text is rendered as plain text (no KaTeX, no HTML); it can be tens of thousands of characters.
- **Daily Double:** splash → wager → clue → verdict. **Final:** subject → wagers hidden → reveal lowest score first (response, verdict, wager, new score).
- **End screen:** standings and a per-model table: buzz accuracy (correct buzzes / buzzes), sealed score (Σ ±value over all clue responses as if every model had buzzed), accuracy over all clues, mean confidence when right vs. wrong, and failures.
- **Controls:** play/pause (space), previous/next clue (←/→), speed 1×/2×/4×, jump to round; clicking a played cell replays that clue.
- **Look:** deep blue board, gold values, bold sans-serif (Google Fonts). No official Jeopardy logo, font, music or theme.
- **Responsive:** at ≤640px, podiums stack under the board, clue text gets smaller, no horizontal page scroll. Colours as CSS tokens on `:root`.

## 9. Benchmark statistics (`stats.py`)

Input: one or more game logs. Output per model (stdout table plus `stats.csv`):
- Clues seen, accuracy (all judged replies), buzz rate, buzz accuracy
- Game score (mean), sealed score (mean), wins
- Calibration: expected calibration error over 10 confidence bins; mean confidence when right vs. wrong
- Wager efficiency: mean DD/Final wager as a fraction of the maximum, and the win rate on those wagers
- Failures and total tokens

Logs are combined across games; no significance testing in v1. The output notes the number of games so readers can judge how noisy it is.

## 10. Testing

- **Unit (pytest):** board generation (deterministic for a seed, no repeated subjects, DD placement rules, Final picks an unused subject); buzz ordering and tie-breaks; scoring for regular, DD and Final clues; wager clamping; JSON parsing, retry and fallback; control rules; stats calculations on a hand-made log.
- **Integration:** `run.py --mock --seed 1` on the fixture produces a log that passes a schema check; `--short` works.
- **Viewer:** Playwright (system Chromium at `/opt/pw-browsers`) loads a mock log, steps through to the end, checks for no console errors, and saves screenshots of the board, clue card, DD, Final and end screen at 1280×800 and 390×844.
- **Live check:** after setup, one `--short` game against the real API with cheap models, inspected by hand.

## 11. Security and cost

- Secrets (`SURPLUS_API_KEY`, `HF_TOKEN`) are read from the environment, falling back to the project `.env` (chmod 600, git-ignored; the Surplus key was copied from the local Hermes config). They are never logged, put in game logs, or written to `config.yaml`. `config.yaml` is git-ignored; `config.example.yaml` is committed.
- All model output (responses, quips, explanations, reasoning) is displayed as escaped text in the viewer.
- A full game is about 400 API calls (contestants plus judge). Token usage per player and for the judge is summed into the log and printed at the end of a run.

## 12. Future (not v1)

Image questions for vision models; clue values by measured difficulty; a games index page; a leaderboard across many seeds; human player mode.
