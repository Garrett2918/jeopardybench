"""Aggregate benchmark metrics over one or more game logs. Spec §9.

    python stats.py games/*.json [--csv stats.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BINS = 10


def load(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for p in paths if not p.name.endswith(".reasoning.json")]


def ece(pairs: list[tuple[float, bool]]) -> float | None:
    """Expected calibration error over BINS equal-width confidence bins."""
    if not pairs:
        return None
    bins = defaultdict(list)
    for conf, ok in pairs:
        bins[min(BINS - 1, int(conf * BINS))].append((conf, ok))
    return sum(len(b) / len(pairs) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b))
               for b in bins.values())


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def compute(logs: list[dict]) -> list[dict]:
    acc = defaultdict(lambda: defaultdict(list))
    for log in logs:
        models = [p["model"] for p in log["players"]]
        sealed = defaultdict(int)
        for e in log["events"]:
            if e["type"] not in ("clue", "final") or e.get("skipped"):
                continue
            for r in e["replies"]:
                m = acc[models[r["player"]]]
                ok = r["verdict"]["correct"]
                m["correct"].append(ok)
                if not r["error"]:
                    m["calib"].append((r["confidence"] / 100, ok))
                if e["type"] == "clue":
                    sealed[r["player"]] += e["value"] if ok else -e["value"]
                    if not e["daily_double"]:
                        m["buzz"].append(r["buzz"])
                        if r["buzz"]:
                            m["buzz_correct"].append(ok)
            if e["type"] == "clue" and e["daily_double"]:
                p, dd = e["dd"]["player"], e["dd"]
                acc[models[p]]["wager_frac"].append(e["wager"] / dd["max_wager"])
                acc[models[p]]["wager_won"].append(e["replies"][p]["verdict"]["correct"])
            if e["type"] == "final":
                for i in e["qualifiers"]:
                    acc[models[i]]["wager_frac"].append(e["wagers"][i] / e["scores_before"][i])
                    acc[models[i]]["wager_won"].append(e["replies"][i]["verdict"]["correct"])
        for i, model in enumerate(models):
            m = acc[model]
            m["games"].append(1)
            m["score"].append(log["result"]["scores"][i])
            m["sealed"].append(sealed[i])
            m["wins"].append(i in log["result"]["winners"])
            u = log["usage"]["per_player"][i]
            m["failures"].append(u["failures"])
            m["tokens"].append(u["tokens_in"] + u["tokens_out"])

    rows = []
    for model, m in acc.items():
        right = [c for c, ok in m["calib"] if ok]
        wrong = [c for c, ok in m["calib"] if not ok]
        rows.append({
            "model": model,
            "games": len(m["games"]),
            "wins": sum(m["wins"]),
            "clues": len(m["correct"]),
            "accuracy": mean(m["correct"]),
            "buzz_rate": mean(m["buzz"]),
            "buzz_accuracy": mean(m["buzz_correct"]),
            "mean_score": mean(m["score"]),
            "mean_sealed_score": mean(m["sealed"]),
            "ece": ece(m["calib"]),
            "conf_when_right": mean(right) * 100 if right else None,
            "conf_when_wrong": mean(wrong) * 100 if wrong else None,
            "wager_frac": mean(m["wager_frac"]),
            "wager_win_rate": mean(m["wager_won"]),
            "failures": sum(m["failures"]),
            "tokens": sum(m["tokens"]),
        })
    return sorted(rows, key=lambda r: -(r["mean_score"] or 0))


def fmt(v, key: str) -> str:
    if v is None:
        return "—"
    if key in ("accuracy", "buzz_rate", "buzz_accuracy", "wager_frac", "wager_win_rate"):
        return f"{v:.0%}"
    if key == "ece":
        return f"{v:.3f}"
    if key in ("mean_score", "mean_sealed_score"):
        return f"{'-' if v < 0 else ''}${abs(v):,.0f}"
    if key.startswith("conf_"):
        return f"{v:.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=Path("stats.csv"))
    args = ap.parse_args(argv)
    logs = load(args.logs)
    rows = compute(logs)
    if not rows:
        raise SystemExit("no game logs given")
    keys = list(rows[0])
    table = [keys] + [[fmt(r[k], k) for k in keys] for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(keys))]
    for j, row in enumerate(table):
        print("  ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(row, widths))))
        if j == 0:
            print("  ".join("-" * w for w in widths))
    print(f"\n{len(logs)} game(s). With few games these numbers are noisy; treat differences with caution.")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
