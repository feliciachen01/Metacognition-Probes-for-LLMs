"""Step 4: results.md plus two plots (probe quality by layer; risk-coverage on the test split)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import risk_coverage


def _cost_lines(run_dir: Path) -> list[str]:
    ans = pd.read_json(run_dir / "answers.jsonl", lines=True)
    per_q = (ans["batch_seconds"] / ans["batch_size"]).mean()
    lines = [f"- Answer generation: {per_q:.3f} s per question (batched, includes saving hidden states)"]
    b = run_dir / "baselines.jsonl"
    if b.exists():
        base = pd.read_json(b, lines=True)
        lines.append(f"- Stated confidence: +{base['stated_seconds'].mean():.3f} s per question (extra call)")
        lines.append(f"- Sampling agreement: +{base['sampling_seconds'].mean():.3f} s per question (extra samples)")
    bench = run_dir / "bench.json"
    if bench.exists():
        bj = json.loads(bench.read_text())
        lines.append(f"- Probe overhead (single request, `kwik bench`): +{bj['added_ms_p50']:.1f} ms p50, "
                     f"+{bj['added_ms_p95']:.1f} ms p95 on {bj['device']}")
    return lines


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    res = pd.read_csv(run_dir / "results.csv")
    sweep = pd.read_csv(run_dir / "layer_sweep.csv")
    scores = pd.read_parquet(run_dir / "scores.parquet")

    # plot 1: validation AUROC by layer (best C per layer)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for position, label in (("pre", "before answer"), ("end", "after answer")):
        s = sweep[sweep.position == position].groupby("layer")["val_auroc"].max()
        ax.plot(s.index, s.values, marker="o", ms=3, label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel("validation AUROC")
    ax.set_title("Where in the model is 'I'm about to be wrong' readable?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "layer_sweep.png", dpi=160)
    plt.close(fig)

    # plot 2: risk-coverage on test
    test = scores[scores.split == "test"]
    methods = [c for c in scores.columns if c not in ("uid", "split", "y")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in methods:
        ok = np.isfinite(test[m].to_numpy())
        if ok.sum() < 2:
            continue
        cov, risk = risk_coverage(test["y"].to_numpy()[ok], test[m].to_numpy()[ok])
        ax.plot(cov, risk, label=m, lw=1.4)
    ax.set_xlabel("share of questions answered (least risky first)")
    ax.set_ylabel("error rate among answered")
    ax.set_title("Risk-coverage, test split")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "risk_coverage.png", dpi=160)
    plt.close(fig)

    def table(split: str) -> str:
        t = res[res.split == split].sort_values("auroc", ascending=False)
        if t.empty:
            return "_no rows_"
        cols = ["method", "n", "auroc", "ece", "brier", "aurc", "acc@80%"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in t.iterrows():
            lines.append("| " + " | ".join([r["method"], str(int(r["n"]))] +
                                           [f"{r[c]:.3f}" for c in cols[2:]]) + " |")
        return "\n".join(lines)

    md = [
        f"# Results: {cfg['run_name']}",
        "",
        f"Model: `{cfg['model_id']}`. Questions: {summary['n']}. "
        f"Error rate by split: { {k: round(v, 3) for k, v in summary['error_rate'].items() if v is not None} }.",
        "",
        "Higher AUROC is better (spotting wrong answers). Lower ECE, Brier and AURC are better.",
        "acc@80% = accuracy if the system answers only the 80% of questions it is least worried about.",
        "",
        "## Test split (same datasets as training)",
        "",
        table("test"),
        "",
        "## Transfer split (dataset never seen in training)",
        "",
        table("transfer"),
        "",
        "## Cost",
        "",
        *_cost_lines(run_dir),
        "",
        "## Chosen probes (picked on validation only)",
        "",
        *[f"- {k}: layer {v['layer']}, C={v['C']}, val AUROC {v['val_auroc']:.3f}" for k, v in summary["choices"].items()],
        "",
        f"Stated-confidence replies that contained a usable number: {summary.get('stated_parse_rate')}",
        "",
        "![layer sweep](layer_sweep.png)",
        "",
        "![risk coverage](risk_coverage.png)",
        "",
        "Automatic correctness labels must be checked against hand labels (`kwik handcheck-agree`) before these numbers are quoted.",
    ]
    out = run_dir / "results.md"
    out.write_text("\n".join(md) + "\n")
    return out
