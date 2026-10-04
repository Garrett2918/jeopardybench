"""Generate tests/fixtures/mini_hle.parquet: a small synthetic HLE-shaped dataset.

Every answer is a short string so the mock judge can compare exactly.
Run: .venv/bin/python tests/make_fixture.py
"""
from pathlib import Path

import pandas as pd

SUBJECTS = {
    "Chess": "Other", "Genetics": "Biology/Medicine", "Law": "Humanities/Social Science",
    "Classical Ballet": "Other", "Physics": "Physics", "Musicology": "Humanities/Social Science",
    "Mathematics": "Math", "Linguistics": "Humanities/Social Science",
    "Computer Science": "Computer Science/AI", "Chemistry": "Chemistry",
    "Economics": "Humanities/Social Science", "Trivia": "Other", "Ecology": "Biology/Medicine",
}


def build() -> pd.DataFrame:
    rows = []
    for s_i, (subject, category) in enumerate(SUBJECTS.items()):
        for k in range(7):
            mc = k % 3 == 2
            q = f"[{subject} #{k}] What is the value of $x$ when $x^2 = {(k + 2) ** 2}$ and $x > 0$?"
            if mc:
                q += "\n\nAnswer Choices:\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5"
            rows.append({
                "id": f"fx{s_i:02d}{k}",
                "question": q * (1 + (k == 6) * 6),  # one long question per subject
                "image": "",
                "image_preview": None,
                "answer": "C" if mc else str(k + 2),
                "answer_type": "multipleChoice" if mc else "exactMatch",
                "author_name": "fixture",
                "rationale": f"Because $({k + 2})^2 = {(k + 2) ** 2}$.",
                "rationale_image": None,
                "raw_subject": subject,
                "category": category,
                "canary": "",
            })
    # an image question and a too-small subject, both of which must be filtered out
    rows.append({**rows[0], "id": "fximg", "image": "data:image/png;base64,AAAA"})
    for k in range(3):
        rows.append({**rows[0], "id": f"fxsmall{k}", "raw_subject": "Tiny Subject"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures" / "mini_hle.parquet"
    out.parent.mkdir(exist_ok=True)
    build().to_parquet(out)
    print(f"wrote {out}")
