"""Check the automatic correctness labels against your own judgement.

export: writes handcheck.csv with 200 random answers and an empty `human_correct` column (1 / 0).
agree:  after you fill it in, reports agreement and Cohen's kappa, and lists disagreements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def export(run_dir: Path, n: int = 200, seed: int = 0) -> Path:
    ans = pd.read_json(Path(run_dir) / "answers.jsonl", lines=True)
    sample = ans.sample(n=min(n, len(ans)), random_state=seed)
    out = sample[["uid", "dataset", "question", "answers", "prediction", "correct"]].copy()
    out["answers"] = out["answers"].apply(lambda a: " | ".join(a[:8]))
    out = out.rename(columns={"correct": "auto_correct"})
    out["auto_correct"] = out["auto_correct"].astype(int)
    out["human_correct"] = ""
    path = Path(run_dir) / "handcheck.csv"
    out.to_csv(path, index=False)
    return path


def agree(run_dir: Path) -> dict:
    df = pd.read_csv(Path(run_dir) / "handcheck.csv")
    df = df[df["human_correct"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        raise SystemExit("handcheck.csv has no filled human_correct values yet")
    h = df["human_correct"].astype(float).astype(int)
    a = df["auto_correct"].astype(int)
    result = {"n": int(len(df)), "agreement": float((h == a).mean()), "kappa": float(cohen_kappa_score(h, a)),
              "auto_says_correct_human_says_wrong": int(((a == 1) & (h == 0)).sum()),
              "auto_says_wrong_human_says_correct": int(((a == 0) & (h == 1)).sum())}
    df[h != a].to_csv(Path(run_dir) / "handcheck_disagreements.csv", index=False)
    return result
