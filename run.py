"""Play one JeopardyBench game and write games/<id>.json (+ .reasoning.json).

    python run.py --seed 42            # real game via the API in config.yaml
    python run.py --seed 1 --short     # Jeopardy round + Final only
    python run.py --seed 1 --mock      # offline mock players on the test fixture
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from jeopardybench import data
from jeopardybench.board import build_game_board
from jeopardybench.env import ROOT, load_dotenv
from jeopardybench.game import Game
from jeopardybench.judge import Judge
from jeopardybench.log import GameLog, Player as PlayerInfo
from jeopardybench.mock import MockJudge, MockPlayer
from jeopardybench.players import OpenAITransport, Player
from jeopardybench.prompts import JUDGE_PROMPT_VERSION

FIXTURE = ROOT / "tests" / "fixtures" / "mini_hle.parquet"


def money(v: int) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,}"


def build_players(cfg: dict, questions, mock: bool, seed: int):
    native = set(cfg.get("native_reasoning") or [])
    if mock:
        answers = questions["answer"].to_dict()
        players = [MockPlayer(c["name"], c["model"], c["color"], answers, skill=0.3 + 0.12 * i,
                              seed=seed * 100 + i, native_reasoning=c["model"] in native)
                   for i, c in enumerate(cfg["contestants"])]
        return players, MockJudge()
    key = os.environ.get(cfg["api"]["api_key_env"])
    if not key:
        sys.exit(f"{cfg['api']['api_key_env']} is not set (env or .env).")
    transport = OpenAITransport(cfg["api"]["base_url"], key, cfg.get("timeout_s", 600))
    players = [Player(c["name"], c["model"], c["color"], transport, c.get("params"), c["model"] in native)
               for c in cfg["contestants"]]
    judge_api = cfg["judge"].get("api")  # optional: the judge may use a different endpoint
    judge_transport = transport
    if judge_api:
        judge_key = os.environ.get(judge_api["api_key_env"])
        if not judge_key:
            sys.exit(f"{judge_api['api_key_env']} is not set (env or .env).")
        judge_transport = OpenAITransport(judge_api["base_url"], judge_key, cfg.get("timeout_s", 600))
    judge = Judge(Player("Judge", cfg["judge"]["model"], "#888", judge_transport, cfg["judge"].get("params")))
    return players, judge


def preflight(players: list, judge) -> None:
    targets = players + ([judge.player] if isinstance(judge, Judge) else [])
    with ThreadPoolExecutor(len(targets)) as ex:
        results = list(ex.map(lambda p: p.preflight(), targets))
    failed = [(p, r) for p, r in zip(targets, results) if r.data is None]
    for p, r in zip(targets, results):
        print(f"  preflight {p.model:24} {'ok' if r.data else 'FAILED: ' + str(r.error)} ({r.latency_ms / 1000:.1f}s)")
    if failed:
        sys.exit("Preflight failed; no game was played.")


def printer(names: list[str]):
    def on_event(e: dict) -> None:
        t = e["type"]
        if t == "round_start":
            print(f"\n=== Round {e['round'] + 1} — {names[e['control']]} has control ===")
        elif t == "pick":
            print(f"{names[e['player']]} picks {e['subject']} for {money(e['value'])}"
                  + (" (fallback)" if e["fallback"] else ""), flush=True)
        elif t == "clue":
            marks = []
            for r in e["replies"]:
                tag = "✓" if r["verdict"]["correct"] else "✗"
                buzz = "B" if r["buzz"] else "-"
                err = " ⚠" if r["error"] else ""
                marks.append(f"{names[r['player']]} {tag}{buzz}{r['confidence']}{err}")
            dd = f" DAILY DOUBLE wager {money(e['wager'])}" if e["daily_double"] else ""
            print(f"  {dd} [{' | '.join(marks)}]  scores: "
                  + ", ".join(f"{n} {money(s)}" for n, s in zip(names, e["scores_after"])), flush=True)
        elif t == "final":
            print(f"\n=== Final Jeopardy: {e['subject']} ===")
            if e["skipped"]:
                print("  skipped: nobody has a positive score")
            else:
                for i in e["qualifiers"]:
                    r = e["replies"][i]
                    print(f"  {names[i]} wagers {money(e['wagers'][i])}: {r['response']!r} "
                          f"{'✓' if r['verdict']['correct'] else '✗'}")
    return on_event


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--short", action="store_true", help="Jeopardy round + Final only")
    ap.add_argument("--mock", action="store_true", help="offline mock players and judge")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=None, help="parquet file (default: HLE, or fixture with --mock)")
    ap.add_argument("--out", type=Path, default=ROOT / "games")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg_path = args.config or (ROOT / "config.yaml" if (ROOT / "config.yaml").exists() else ROOT / "config.example.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)

    path = args.data or (FIXTURE if args.mock else data.download_if_missing())
    questions = data.load_questions(path)
    rng = random.Random(seed)
    board = build_game_board(questions, rng, short=args.short)
    players, judge = build_players(cfg, questions, args.mock, seed)

    print(f"JeopardyBench seed={seed} {'(mock) ' if args.mock else ''}"
          f"{' vs '.join(p.model for p in players)} | judge {judge.model}")
    if not args.mock and not args.skip_preflight:
        preflight(players, judge)

    started = time.time()
    names = [p.name for p in players]
    game = Game(board, questions, players, judge, rng, on_event=printer(names))
    result = game.play()

    used = [c.qid for r in board.rounds for c in r.cells] + [board.final["qid"]]
    cols = ["question", "answer", "answer_type", "rationale", "category", "raw_subject"]
    game_id = f"{datetime.now():%Y-%m-%dT%H-%M-%S}_seed{seed}{'_mock' if args.mock else ''}"
    log = GameLog(
        id=game_id, seed=seed,
        config={k: v for k, v in cfg.items() if k != "api"} | {"base_url": cfg["api"]["base_url"],
                                                              "short": args.short, "mock": args.mock,
                                                              "judge_prompt_version": JUDGE_PROMPT_VERSION},
        dataset={"file": path.name, "sha256": data.sha256(path)},
        players=[PlayerInfo(p.name, p.model, p.color) for p in players],
        rounds=board.rounds, final=board.final,
        questions={q: {c: str(questions.at[q, c]) for c in cols} for q in used},
        events=game.events, result=result,
        usage={"per_player": [dict(p.usage) for p in players], "judge": dict(judge.usage),
               "duration_s": round(time.time() - started, 1)},
    )
    n = 2
    while (args.out / f"{log.id}.json").exists():
        log.id = f"{game_id}-{n}"
        n += 1
    game_id = log.id
    out = args.out / f"{game_id}.json"
    log.save(out)
    (args.out / f"{game_id}.reasoning.json").write_text(json.dumps(game.reasoning, ensure_ascii=False))

    print("\n=== Result ===")
    for i, p in enumerate(players):
        u = p.usage
        star = " 🏆" if i in result["winners"] else ""
        print(f"  {p.name:10} {money(result['scores'][i]):>9}{star}   calls {u['calls']}, "
              f"tokens in {u['tokens_in']:,} / out {u['tokens_out']:,}, failures {u['failures']}")
    print(f"  judge: calls {judge.usage['calls']}, tokens in {judge.usage['tokens_in']:,} / out {judge.usage['tokens_out']:,}")
    print(f"  time {log.usage['duration_s'] / 60:.1f} min → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
