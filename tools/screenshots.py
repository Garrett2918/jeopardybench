"""Regenerate the README screenshots in docs/images/.

    .venv/bin/python tools/screenshots.py --real games/<id>.json

Board, results and phone shots come from the real game you pass (they show only subjects
and stats). Clue shots come from a mock game, with one clue replaced by an invented
HLE-style question, so no real HLE question is ever published (HLE asks for that).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402

OUT = ROOT / "docs" / "images"

# Invented for these screenshots; not from HLE. Answer checked numerically: trace(A^4) = 1482.
SHOWCASE = {
    "question": (
        "Let $p = 13$ and let $G = \\mathrm{Cay}(\\mathbb{Z}_p, S)$ be the Cayley graph on the additive group "
        "$\\mathbb{Z}_p$ with connection set $S = \\{x^2 : x \\in \\mathbb{Z}_p^{\\times}\\}$, so that $i \\sim j$ "
        "if and only if $i - j$ is a nonzero quadratic residue modulo $p$. Let $A$ be the adjacency matrix of $G$ "
        "and, for $\\ell \\ge 1$, let $W_\\ell = \\operatorname{tr}(A^\\ell)$ denote the number of closed walks of "
        "length $\\ell$ in $G$, where walks may revisit vertices and edges, and two walks that start at different "
        "vertices or traverse the same cycle in opposite directions are counted separately.\n\n"
        "It is known that $G$ is strongly regular. Without computing the spectrum of $A$ explicitly, "
        "determine $W_4$. Give your answer as an integer."
    ),
    "answer": "1482",
    "answer_type": "exactMatch",
    "category": "Math",
    "rationale": (
        "Since $-1$ is a square mod 13, $S = -S$ and $G$ is the Paley graph $P(13)$, strongly regular with "
        "parameters $(13, 6, 2, 3)$. Then $(A^2)_{ii} = 6$, $(A^2)_{ij} = 2$ for the 6 neighbours of $i$ and $3$ "
        "for the 6 non-neighbours. The closed 4-walks at $i$ number $\\sum_j (A^2)_{ij}^2 = 36 + 6 \\cdot 4 + 6 \\cdot 9 = 114$, "
        "so $W_4 = 13 \\cdot 114 = 1482$."
    ),
}
RIGHT = {
    "response": "What is 1482?",
    "explanation": "P(13) is strongly regular with parameters (13, 6, 2, 3), so each row of A² holds one 6, six 2s "
                   "and six 3s. Squaring and summing gives 114 closed 4-walks per vertex, and 13 · 114 = 1482.",
    "quip": "Strongly regular graphs do the counting for you.",
    "extracted": "1482",
    "reasoning": "The final answer 1482 matches the correct answer exactly.",
}
WRONG = [
    {"response": "What is 1296?",
     "explanation": "Every vertex has degree 6, and the dominant eigenvalue is 6, so the trace of A⁴ is 6⁴ = 1296.",
     "quip": "Six to the fourth. Final answer.",
     "extracted": "1296",
     "reasoning": "1296 counts only the contribution of the eigenvalue 6; the correct answer is 1482."},
    {"response": "What is 156?",
     "explanation": "Each edge lies in λ = 2 triangles; with 39 edges that gives 26 triangles and 156 closed walks.",
     "quip": "Triangles all the way down.",
     "extracted": "156",
     "reasoning": "156 is the number of closed walks of length 3, not 4; the correct answer is 1482."},
]
THINKING = {
    True: "Connection set: squares mod 13 are {1, 3, 4, 9, 10, 12}. Since 12 = -1 is a square, S = -S, "
          "so the graph is undirected: the Paley graph P(13).\nParameters: k = 6. Adjacent vertices share "
          "λ = (q - 5)/4 = 2 neighbours; non-adjacent share μ = (q - 1)/4 = 3.\ntr(A^4) = Σ_i Σ_j (A²)_ij².\n"
          "Row of A²: diagonal 6, six entries 2, six entries 3.\nΣ = 36 + 6·4 + 6·9 = 36 + 24 + 54 = 114.\n"
          "W4 = 13 · 114 = 1482. Check with the spectrum: eigenvalues 6 and (-1 ± √13)/2 each with "
          "multiplicity 6; 6⁴ + 6·31 = 1296 + 186 = 1482. Consistent.",
    False: "Degree 6, so the largest eigenvalue is 6. The other eigenvalues should be small; tr(A^4) is "
           "dominated by 6^4 = 1296. I'm fairly sure the remaining terms are negligible here.",
}


def make_mock(tmp: Path) -> Path:
    """Play mock games until one has a Mathematics clue with a wrong buzz before a right one."""
    for seed in range(1, 200):
        out = tmp / f"mock{seed}"
        run.main(["--mock", "--seed", str(seed), "--out", str(out)])
        log_path = next(p for p in out.glob("*.json") if not p.name.endswith(".reasoning.json"))
        log = json.loads(log_path.read_text())
        for ei, e in enumerate(log["events"]):
            if (e["type"] == "clue" and e["subject"] == "Mathematics" and not e["daily_double"]
                    and len(e["buzz_order"]) >= 2 and e["value"] >= 600):
                ok = [e["replies"][p]["verdict"]["correct"] for p in e["buzz_order"]]
                if not ok[0] and any(ok):
                    patch(log, ei, log_path)
                    return log_path
    raise SystemExit("no suitable mock game found")


def patch(log: dict, ei: int, log_path: Path) -> None:
    e = log["events"][ei]
    log["questions"][e["qid"]].update({k: SHOWCASE[k] for k in ("question", "answer", "answer_type", "category", "rationale")})
    side_path = log_path.with_name(log_path.stem + ".reasoning.json")
    side = json.loads(side_path.read_text())
    wrong = iter(WRONG * 3)
    right_quips = iter(["Strongly regular graphs do the counting for you.", "Paley graphs: never not 13.",
                        "Trace it twice, buzz once."])
    for r in e["replies"]:
        src = RIGHT if r["verdict"]["correct"] else next(wrong)
        quip = next(right_quips) if r["verdict"]["correct"] else src["quip"]
        r.update(response=src["response"], explanation=src["explanation"], quip=quip, error=None)
        r["verdict"].update(extracted_answer=src["extracted"], reasoning=src["reasoning"],
                            method="judge x2", votes=[r["verdict"]["correct"]] * 2)
        side[f"{ei}:{r['player']}"] = {"kind": "native", "text": THINKING[r["verdict"]["correct"]]}
    log_path.write_text(json.dumps(log))
    side_path.write_text(json.dumps(side))
    log["_showcase"] = ei


def chromium() -> str | None:
    if os.environ.get("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    found = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
    return str(found[-1]) if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, help="a real game log for the board, results and phone shots")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        site = tmp / "site"
        for name in ("mock", "real"):
            (site / name).mkdir(parents=True)
            shutil.copy(ROOT / "viewer.html", site / name)
        mock_log = make_mock(tmp)
        shutil.copy(mock_log, site / "mock" / "game.json")
        shutil.copy(mock_log.with_name(mock_log.stem + ".reasoning.json"), site / "mock" / "game.reasoning.json")
        if args.real:
            shutil.copy(args.real, site / "real" / "game.json")

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
        handler.log_message = lambda *a: None
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"
        OUT.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            exe = chromium()
            browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()

            def page(name, w=1280, h=800):
                pg = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 600 else 1)
                pg.goto(f"{base}/{name}/viewer.html?paused")
                pg.wait_for_function("window.JB && JB.state.steps.length > 0")
                pg.wait_for_timeout(800)
                return pg

            def goto(pg, pick):
                idx = pg.evaluate(f"JB.state.steps.findIndex({pick})")
                pg.evaluate(f"JB.goto({idx})")
                pg.wait_for_timeout(1500)

            showcase = "s => s.clue && JB.state.log.questions[s.clue.e.qid].answer === '1482'"
            pg = page("mock")
            goto(pg, f"s => s.kind === 'buzz' && ({showcase})(s)")
            pg.screenshot(path=str(OUT / "question.png"), full_page=True)
            goto(pg, f"s => s.kind === 'reveal' && ({showcase})(s)")
            pg.locator("details.think").first.evaluate("d => d.open = true")
            pg.wait_for_timeout(600)
            pg.screenshot(path=str(OUT / "reveal.png"), full_page=True)

            if args.real:
                pg = page("real")
                picks = "s => s.kind === 'pick'"
                idx = pg.evaluate(f"JB.state.steps.map((s, i) => [s, i]).filter(([s]) => ({picks})(s)).map(([, i]) => i)[13]")
                pg.evaluate(f"JB.goto({idx})")
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(OUT / "board.png"))
                goto(pg, "s => s.kind === 'end'")
                pg.screenshot(path=str(OUT / "results.png"))
                pg = page("real", 390, 844)
                pg.evaluate(f"JB.goto({idx})")
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(OUT / "phone.png"))
            browser.close()
        httpd.shutdown()
    print(f"screenshots written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
