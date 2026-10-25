"""Step 1: run the model on every question, score the answers, and save hidden states.

Writes into runs/<run_name>/:
  questions.jsonl        every sampled question with its split (train / val / test / transfer)
  shards/00000.npz ...   hidden states per batch ("pre" and "end" positions), resumable
  answers.jsonl          one row per question: prediction, correctness, token log-probs, timing
Re-running the command skips questions that already have a shard, so a Colab disconnect costs one batch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import Config
from .data import load_spec
from .model import generate, load_model
from .prompts import answer_messages, render
from .scoring import clean_prediction, score


def assign_splits(rows: list[dict], fractions: dict, seed: int) -> list[dict]:
    """Stratified by dataset, so every split has the same dataset mix. Transfer sets stay whole."""
    rng = random.Random(seed)
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds_rows in by_ds.values():
        if ds_rows[0]["role"] == "transfer":
            for r in ds_rows:
                r["split"] = "transfer"
            continue
        rng.shuffle(ds_rows)
        n = len(ds_rows)
        n_train = int(round(fractions["train"] * n))
        n_val = int(round(fractions["val"] * n))
        for i, r in enumerate(ds_rows):
            r["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return rows


def prepare_questions(cfg: Config) -> list[dict]:
    path = cfg.run_dir / "questions.jsonl"
    if path.exists():  # keep the original sample and splits on resume
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows: list[dict] = []
    for i, spec in enumerate(cfg.datasets):
        rows.extend(load_spec(spec, seed=cfg.seed + i))
    rows = assign_splits(rows, cfg.split, cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return rows


def done_uids(run_dir: Path) -> set[str]:
    path = run_dir / "answers.jsonl"
    if not path.exists():
        return set()
    return {json.loads(l)["uid"] for l in path.read_text().splitlines() if l.strip()}


def collect(cfg: Config, model=None, tok=None) -> Path:
    rows = prepare_questions(cfg)
    run_dir = cfg.run_dir
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    finished = done_uids(run_dir)
    todo = [r for r in rows if r["uid"] not in finished]
    print(f"[collect] {len(rows)} questions, {len(finished)} done, {len(todo)} to go")
    if not todo:
        return run_dir
    if model is None:
        model, tok, device = load_model(cfg.model_id, cfg.device, cfg.dtype, cfg.load_in_4bit)
        print(f"[collect] loaded {cfg.model_id} on {device}")

    next_shard = len(list(shard_dir.glob("*.npz")))
    with open(run_dir / "answers.jsonl", "a") as fout:
        for start in range(0, len(todo), cfg.batch_size):
            batch = todo[start : start + cfg.batch_size]
            prompts = [render(tok, answer_messages(r["question"])) for r in batch]
            g = generate(model, tok, prompts, cfg.max_new_tokens, capture=True, layer_stride=cfg.layer_stride)
            np.savez_compressed(
                shard_dir / f"{next_shard:05d}.npz",
                uids=np.array([r["uid"] for r in batch]),
                pre=g.pre.astype(np.float16), end=g.end.astype(np.float16), layers=np.array(g.layers),
            )
            next_shard += 1
            for i, r in enumerate(batch):
                pred = clean_prediction(g.texts[i])
                s = score(pred, r["answers"])
                fout.write(json.dumps({
                    "uid": r["uid"], "dataset": r["dataset"], "split": r["split"], "question": r["question"],
                    "answers": r["answers"][:20], "raw": g.texts[i], "prediction": pred, **s,
                    "logprob_mean": float(g.logprob_mean[i]), "logprob_min": float(g.logprob_min[i]),
                    "n_tokens": int(g.n_tokens[i]), "batch_seconds": g.seconds, "batch_size": len(batch),
                }) + "\n")
            fout.flush()
            done = len(finished) + start + len(batch)
            print(f"[collect] {done}/{len(rows)}  ({g.seconds / len(batch):.2f}s per question)")
    return run_dir


def load_features(run_dir: Path) -> tuple[list[str], dict[str, np.ndarray], list[int]]:
    """Concatenate shards -> uids, {"pre": (N, L, D), "end": (N, L, D)}, layer indices."""
    uids, pre, end, layers = [], [], [], None
    for f in sorted((run_dir / "shards").glob("*.npz")):
        z = np.load(f)
        uids.extend(z["uids"].tolist())
        pre.append(z["pre"])
        end.append(z["end"])
        layers = z["layers"].tolist()
    return uids, {"pre": np.concatenate(pre), "end": np.concatenate(end)}, layers
