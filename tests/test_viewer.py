"""Viewer smoke test: load a mock game, walk every step, screenshot key screens.

Screenshots go to tests/screenshots/ (git-ignored) for review by eye.
"""
import functools
import http.server
import json
import os
import threading
from pathlib import Path

import pytest

import run

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "tests" / "screenshots"
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    games = tmp_path_factory.mktemp("games")
    run.main(["--mock", "--seed", "11", "--out", str(games)])
    log = next(p for p in games.glob("*.json") if "reasoning" not in p.name)
    # inject a hostile response to prove model text is never rendered as HTML
    data = json.loads(log.read_text())
    clue = next(e for e in data["events"] if e["type"] == "clue")
    clue["replies"][0]["response"] = '<img src=x onerror="window.__pwned=1">'
    log.write_text(json.dumps(data))
    site = tmp_path_factory.mktemp("site")
    (site / "viewer.html").write_text((ROOT / "viewer.html").read_text())
    (site / "game.json").write_text(log.read_text())
    (site / "game.reasoning.json").write_text(log.with_name(log.stem + ".reasoning.json").read_text())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", data
    httpd.shutdown()


def find_chromium():
    """A pre-installed Chromium, in case it doesn't match the pip playwright's pinned build."""
    if os.environ.get("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    candidates = [Path("/opt/pw-browsers/chromium")]
    candidates += sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"), reverse=True)
    return next((str(c) for c in candidates if c.is_file()), None)


def launch(p):
    try:
        return p.chromium.launch()
    except Exception:
        exe = find_chromium()
        if not exe:
            raise
        return p.chromium.launch(executable_path=exe)


@pytest.mark.parametrize("size", [(1280, 800), (390, 844)])
def test_walkthrough(server, size):
    url, data = server
    SHOTS.mkdir(exist_ok=True)
    tag = f"{size[0]}"
    with playwright.sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"{url}/viewer.html?paused")
        page.wait_for_function("window.JB && JB.state.steps.length > 0")
        n = page.evaluate("JB.state.steps.length")
        kinds = page.evaluate("JB.state.steps.map(s => s.kind)")
        assert kinds[0] == "title" and kinds[-1] == "end"

        def shot(name, index):
            page.evaluate(f"JB.goto({index})")
            page.wait_for_timeout(250)
            page.screenshot(path=str(SHOTS / f"{tag}-{name}.png"), full_page=True)

        for i in range(n):
            page.evaluate(f"JB.goto({i})")
        first_pick = kinds.index("pick")
        picks = [i for i, k in enumerate(kinds) if k == "pick"]
        shot("board", picks[12])
        shot("clue-read", kinds.index("read"))
        shot("buzz", kinds.index("buzz", 40))
        shot("verdict", kinds.index("verdict", 40))
        reveal = kinds.index("reveal")
        shot("reveal", reveal)
        # open a thinking panel: loads the sidecar lazily, renders as text
        page.locator("details.think").first.evaluate("d => d.open = true")
        page.wait_for_timeout(300)
        body = page.locator("details.think .body").first.inner_text()
        assert "<script>" in body  # mock reasoning contains a literal tag, shown as text
        page.screenshot(path=str(SHOTS / f"{tag}-thinking.png"), full_page=True)
        shot("daily-double", kinds.index("ddSplash"))
        shot("dd-wager", kinds.index("wager"))
        shot("final", kinds.index("finalIntro"))
        if "finalWagers" in kinds:
            shot("final-verdict", kinds.index("verdict", kinds.index("finalWagers")))
        shot("end", n - 1)

        # scores on the end screen match the log
        end_scores = page.evaluate("JB.state.steps.at(-1).scores")
        assert end_scores == data["result"]["scores"]
        assert page.evaluate("window.__pwned") is None
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"horizontal overflow {overflow}px"
        # playback actually advances
        page.evaluate("JB.goto(0)")
        page.click("#play")
        page.click("[data-speed='4']")
        page.wait_for_timeout(1500)
        assert page.evaluate("JB.state.i") > 0
        browser.close()
    assert not [e for e in errors if "favicon" not in e], errors
