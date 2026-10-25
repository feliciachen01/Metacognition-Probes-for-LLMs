"""Question sets with known answers.

Every loader returns a list of dicts: {uid, dataset, question, answers: list[str]}.
Hub IDs were checked on 2026-10-25:
  - mandarjoshi/trivia_qa, config rc.nocontext, split validation (17.9K rows)
  - google-research-datasets/nq_open, split validation (3.6K rows, CC BY-SA 3.0)
  - akariasai/PopQA, split test (14.3K rows; long-tail entities, used here as the held-out transfer set)
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def _sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_triviaqa(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows = []
    for r in ds:
        aliases = list(dict.fromkeys([r["answer"]["value"], *r["answer"]["aliases"]]))
        rows.append({"uid": f"tqa-{r['question_id']}", "dataset": "triviaqa",
                     "question": r["question"], "answers": aliases})
    return _sample(rows, n, seed)


def load_nq_open(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    rows = [{"uid": f"nq-{i}", "dataset": "nq_open", "question": r["question"].rstrip("?") + "?",
             "answers": list(r["answer"])} for i, r in enumerate(ds)]
    return _sample(rows, n, seed)


def load_popqa(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    rows = [{"uid": f"popqa-{r['id']}", "dataset": "popqa", "question": r["question"],
             "answers": json.loads(r["possible_answers"])} for r in ds]
    return _sample(rows, n, seed)


def load_local(path: str, n: int, seed: int) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if line.strip():
            r = json.loads(line)
            rows.append({"uid": r.get("uid", f"local-{i}"), "dataset": r.get("dataset", "local"),
                         "question": r["question"], "answers": r["answers"]})
    return _sample(rows, n, seed)


LOADERS = {"triviaqa": load_triviaqa, "nq_open": load_nq_open, "popqa": load_popqa}


def load_spec(spec, seed: int) -> list[dict]:
    if spec.name == "local":
        rows = load_local(spec.path, spec.n, seed)
    else:
        rows = LOADERS[spec.name](spec.n, seed)
    for r in rows:
        r["role"] = spec.role
    return rows
