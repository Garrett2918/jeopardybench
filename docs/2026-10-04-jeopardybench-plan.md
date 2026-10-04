# JeopardyBench — Implementation Plan

Spec: `2026-10-04-jeopardybench-design.md`. Tasks in order; each ends green before the next starts.
Environment: Python venv at `.venv` (`openai`, `pandas`, `pyarrow`, `pyyaml`, `pytest`, `playwright`); Chromium at `/opt/pw-browsers`.

| # | Task | Files | Verify |
|---|---|---|---|
| 1 | **Scaffold.** venv, `requirements.txt`, package skeleton, `config.example.yaml` (spec §4 lineup), `config.yaml` copy, `git init` with the existing `.gitignore`. `env.py`: load `.env` without overriding real env vars. | `requirements.txt`, `jeopardybench/__init__.py`, `jeopardybench/env.py`, `config.example.yaml` | `python -c "import jeopardybench"`; `git status` shows `.env` ignored |
| 2 | **Log types.** Dataclasses for the board, events, replies, verdicts, result and usage; `to_json` / `from_json`; version field. | `jeopardybench/log.py`, `tests/test_log.py` | round-trip test passes |
| 3 | **Data.** `download_if_missing()` (HF_TOKEN, cache at `data/hle.parquet`, sha256), `load_questions()` (text-only filter), `eligible_subjects()`. Generate the fixture `tests/fixtures/mini_hle.parquet` (synthetic, 8 subjects × 7–8 questions, a mix of answer types). | `jeopardybench/data.py`, `tests/make_fixture.py`, `tests/test_data.py` | tests pass; real file gives 2,158 text questions / 49 subjects |
| 4 | **Board.** `build_game_board(df, rng, short)`: rounds, values, DD placement (rows 2–5, distinct columns in R2), Final subject unused + exactMatch. | `jeopardybench/board.py`, `tests/test_board.py` | determinism for a seed, no repeated subjects, DD rules, Final rules |
| 5 | **Prompts.** System, clue, DD answer, pick, wager, final wager, final answer, judge, preflight; JSON schemas plus a validate/clamp function per reply type. | `jeopardybench/prompts.py`, `tests/test_prompts.py` | parsing tests: fenced JSON, extra prose, clamping, missing fields → invalid |
| 6 | **Players and judge.** `Player.ask(kind, ctx)` via the `openai` SDK (base_url, timeout, params passthrough); JSON retry; 429/5xx retry; failure fallback; latency, token and reasoning capture (spec §6.1). `Judge.grade()`. `MockPlayer` / `MockJudge` (seeded; mock judge compares against the answer, mock player is right with a per-player probability and reports a noisy confidence). | `jeopardybench/players.py`, `jeopardybench/judge.py`, `jeopardybench/mock.py`, `tests/test_players.py` | tests with a fake HTTP client: retry, fallback, reasoning extraction (`reasoning_content` / `reasoning` / `reasoning_details`) |
| 7 | **Game engine.** State machine per spec §5: control, picks, parallel clue calls (thread pool), buzz order, scoring, DD, Final, uncounted judging, usage totals, reasoning sidecar. | `jeopardybench/game.py`, `tests/test_game.py` | scripted mock players cover: wrong-then-right buzz chain, nobody buzzes, DD win/loss, wager clamping, Final with a ≤0 player sitting out, tie → co-champions, R2 starts with the lowest scorer |
| 8 | **CLI.** `run.py --seed N [--short] [--mock] [--config path] [--out games/]`: preflight (spec §5.6), progress lines per clue, end summary (scores, tokens). Writes `games/<id>.json` and `games/<id>.reasoning.json`. | `run.py`, `tests/test_cli.py` | `python run.py --mock --seed 1` writes a valid log; preflight failure exits non-zero before any game call |
| 9 | **Stats.** Spec §9 metrics from 1..N logs; table plus `stats.csv`. | `stats.py`, `tests/test_stats.py` | hand-made log with known numbers → exact expected metrics |
| 10 | **Viewer: core.** Loading (URL, drop, picker), `buildTimeline` (pure, exported for tests), board, podiums, clue card, KaTeX, step player, controls, keyboard shortcuts. | `viewer.html` | Playwright: loads the mock log, steps to the end with no console errors |
| 11 | **Viewer: show polish.** Buzz reveal animation, verdicts, score tween, DD splash and wager, Final reveal order, "Everyone's answers" with explanations, "Why" panel, 🧠 Thinking panel (lazy sidecar), end-screen stats, responsive layout, light/dark tokens. | `viewer.html`, `tests/test_viewer.py` | Playwright screenshots (board, clue, DD, Final, end) at 1280×800 and 390×844, reviewed by eye; reasoning text is escaped (a `<script>` reply renders as text) |
| 12 | **Live game.** Download the real dataset; `run.py --seed 1 --short` with the flash lineup (DeepSeek v4.1 Flash, GLM 5.3 Flash, MiMo v2.6 Flash; judge gpt-6-astra); inspect the log and replay; fix issues; then a full game `--seed 2`. Save one as `games/examples/`. | — | game finishes; judge verdicts spot-checked against answers by hand (≥10 clues); replay watched end to end |
| 13 | **README.** Setup (HF terms, tokens, `.env`), running, viewing (`python -m http.server` then `viewer.html?game=…`), stats, cost notes. Optionally publish an example replay as a private artifact link. | `README.md` | follow the README from a clean shell |

**Out of scope here:** spec §12 items.

**Risks to watch**
- The judge disagrees with the HLE answer on borderline exact matches. Mitigation: spot-check in task 12; the judge prompt is versioned in the log.
- Flash models time out or ramble past `max_tokens` on long math clues. Mitigation: fallback is a non-buzz, and failures are counted in stats.
- Surplus routes a model to a different upstream mid-game, so behaviour drifts. Mitigation: record per-call `model` and provider fields from the response, if present.

## TODO

- [ ] **Save the log incrementally.** Write `games/<id>.json` (and the reasoning sidecar) after every clue, marked `"complete": false` until the game ends, so a cancelled or crashed game can still be replayed and its API spend isn't lost. The viewer should show a "game incomplete" end screen for such logs. (Found 2026-10-04: a cancelled live game lost ~5 clues of answers.)
- [ ] **Pick a third contestant.** MiMo v2.6 Flash is excluded for now: up to ~8 min per clue at 32k tokens.
