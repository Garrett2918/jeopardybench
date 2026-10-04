"""HLE dataset download, loading and subject eligibility. Spec §3."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

from .env import ROOT

HLE_URL = "https://huggingface.co/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet"
DEFAULT_PATH = ROOT / "data" / "hle.parquet"
COLUMNS = ["id", "question", "image", "answer", "answer_type", "rationale", "raw_subject", "category"]
MIN_PER_SUBJECT = 5


def download_if_missing(path: Path = DEFAULT_PATH) -> Path:
    if path.exists():
        return path
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (env or .env); it is needed to download the gated HLE dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(HLE_URL, headers={"Authorization": f"Bearer {token}"})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_questions(path: Path) -> pd.DataFrame:
    """Text-only questions, indexed by id."""
    df = pd.read_parquet(path, columns=COLUMNS)
    df = df[df["image"].fillna("") == ""].drop(columns=["image"])
    df["rationale"] = df["rationale"].fillna("")
    return df.set_index("id", drop=False)


def eligible_subjects(df: pd.DataFrame, minimum: int = MIN_PER_SUBJECT) -> list[str]:
    counts = df["raw_subject"].value_counts()
    return sorted(counts[counts >= minimum].index)
