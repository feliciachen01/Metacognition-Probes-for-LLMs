"""Step 3: train probes on hidden states and compare them with the baselines.

Label y = 1 when the model's answer is wrong. A probe is StandardScaler + logistic regression on one
layer at one position ("pre" = before the first answer token, "end" = after the last one).
Layer, position and regularisation are chosen on the validation split only; test and transfer are
touched once, at the end.

Every baseline gets a 1-D logistic (Platt) mapping fitted on the validation split, so calibration
metrics compare probabilities with probabilities. AUROC is unaffected by that mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .collect import load_features
from .metrics import auroc, summarize

C_GRID = (0.01, 0.1, 1.0)


@dataclass
class ProbeChoice:
    position: str
    layer: int        # index into hidden_states (0 = embeddings)
    layer_pos: int    # column in the saved array
    C: float
    val_auroc: float


def load_table(run_dir: Path) -> pd.DataFrame:
    ans = pd.read_json(run_dir / "answers.jsonl", lines=True)
    ans["y"] = (~ans["correct"].astype(bool)).astype(int)
    b = run_dir / "baselines.jsonl"
    if b.exists():
        base = pd.read_json(b, lines=True)
        ans = ans.merge(base[["uid", "stated_conf", "sampling_agree", "stated_seconds", "sampling_seconds"]],
                        on="uid", how="left")
    return ans


def fit_probe(X: np.ndarray, y: np.ndarray, C: float):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)
    return clf


def layer_sweep(feats: dict, layers: list[int], idx: dict, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, arr in feats.items():
        for j, layer in enumerate(layers):
            Xtr, Xva = arr[idx["train"], j].astype(np.float32), arr[idx["val"], j].astype(np.float32)
            for C in C_GRID:
                clf = fit_probe(Xtr, y[idx["train"]], C)
                rows.append({"position": position, "layer": layer, "layer_pos": j, "C": C,
                             "val_auroc": auroc(y[idx["val"]], clf.predict_proba(Xva)[:, 1])})
    return pd.DataFrame(rows)


def platt(score_val: np.ndarray, y_val: np.ndarray):
    m = np.isfinite(score_val)
    lr = LogisticRegression(max_iter=1000).fit(score_val[m].reshape(-1, 1), y_val[m])
    return lambda s: lr.predict_proba(np.nan_to_num(s, nan=np.nanmean(score_val)).reshape(-1, 1))[:, 1]


def baseline_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw risk scores, higher = more likely wrong."""
    out = {"logprob (mean)": -df["logprob_mean"].to_numpy(), "logprob (min)": -df["logprob_min"].to_numpy()}
    if "stated_conf" in df:
        out["stated confidence"] = 1 - df["stated_conf"].astype(float).to_numpy()
    if "sampling_agree" in df:
        out["sampling agreement"] = 1 - df["sampling_agree"].astype(float).to_numpy()
    return out


def train_and_evaluate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    df = load_table(run_dir)
    uids, feats, layers = load_features(run_dir)
    order = {u: i for i, u in enumerate(uids)}
    df = df[df["uid"].isin(order)].copy()
    df["row"] = df["uid"].map(order)
    df = df.sort_values("row").reset_index(drop=True)
    feats = {k: v[df["row"].to_numpy()] for k, v in feats.items()}
    y = df["y"].to_numpy()
    idx = {s: np.where(df["split"].to_numpy() == s)[0] for s in ("train", "val", "test", "transfer")}

    sweep = layer_sweep(feats, layers, idx, y)
    sweep.to_csv(run_dir / "layer_sweep.csv", index=False)

    choices = {}
    for position in feats:
        best = sweep[sweep.position == position].sort_values("val_auroc", ascending=False).iloc[0]
        choices[position] = ProbeChoice(position, int(best.layer), int(best.layer_pos), float(best.C),
                                        float(best.val_auroc))

    results, scores_out = [], {}
    for position, ch in choices.items():
        Xtr = feats[position][idx["train"], ch.layer_pos].astype(np.float32)
        clf = fit_probe(Xtr, y[idx["train"]], ch.C)
        joblib.dump({"model": clf, "position": position, "layer": ch.layer, "C": ch.C,
                     "model_id": json.loads((run_dir / "config.json").read_text())["model_id"]},
                    run_dir / f"probe_{position}.joblib")
        name = f"probe ({'before answer' if position == 'pre' else 'after answer'}, layer {ch.layer})"
        risk_all = clf.predict_proba(feats[position][:, ch.layer_pos].astype(np.float32))[:, 1]
        scores_out[name] = risk_all

    for name, raw in baseline_scores(df).items():
        if np.isfinite(raw[idx["val"]]).sum() < 10:
            continue
        cal = platt(raw[idx["val"]], y[idx["val"]])
        scores_out[name] = cal(raw)

    for name, risk in scores_out.items():
        for split in ("test", "transfer"):
            ii = idx[split]
            if len(ii) == 0:
                continue
            ok = np.isfinite(risk[ii])
            if ok.sum() == 0:
                continue
            results.append({"method": name, "split": split, **summarize(y[ii][ok], risk[ii][ok]),
                            "coverage_of_split": float(ok.mean())})

    res = pd.DataFrame(results)
    res.to_csv(run_dir / "results.csv", index=False)
    pd.DataFrame({"uid": df["uid"], "split": df["split"], "y": y, **scores_out}).to_parquet(run_dir / "scores.parquet")
    summary = {"choices": {k: vars(v) for k, v in choices.items()},
               "n": {k: int(len(v)) for k, v in idx.items()},
               "error_rate": {k: float(y[v].mean()) if len(v) else None for k, v in idx.items()},
               "stated_parse_rate": (float(df["stated_conf"].notna()[df["split"] != "train"].mean())
                                     if "stated_conf" in df else None)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
