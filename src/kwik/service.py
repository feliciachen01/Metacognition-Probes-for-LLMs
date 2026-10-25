"""Step 5: the confidence API and its latency benchmark.

POST /answer {"question": "..."} ->
  {"answer", "risk_wrong", "risk_from": "before answer, layer N", "timings_ms": {...}}

The risk score comes from the "pre" probe, so it is known after the prompt pass, before the answer
is written. `bench` measures what the probe costs on top of plain generation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from .model import generate, load_model
from .prompts import answer_messages, render
from .scoring import clean_prediction


class ConfidenceService:
    def __init__(self, run_dir: str | Path, device: str = "auto", max_new_tokens: int = 24):
        run_dir = Path(run_dir)
        bundle = joblib.load(run_dir / "probe_pre.joblib")
        self.probe, self.layer = bundle["model"], bundle["layer"]
        self.model, self.tok, self.device = load_model(bundle["model_id"], device)
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def risk_before_answer(self, prompt: str) -> tuple[float, float]:
        t0 = time.perf_counter()
        enc = self.tok([prompt], return_tensors="pt").to(self.model.device)
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        x = hs[self.layer][0, -1, :].float().cpu().numpy().reshape(1, -1)
        risk = float(self.probe.predict_proba(x)[0, 1])
        return risk, (time.perf_counter() - t0) * 1000

    def answer(self, question: str) -> dict:
        prompt = render(self.tok, answer_messages(question))
        risk, risk_ms = self.risk_before_answer(prompt)
        g = generate(self.model, self.tok, [prompt], self.max_new_tokens, capture=False)
        return {"answer": clean_prediction(g.texts[0]), "risk_wrong": round(risk, 4),
                "risk_from": f"before answer, layer {self.layer}",
                "timings_ms": {"risk": round(risk_ms, 1), "generate": round(g.seconds * 1000, 1)}}


def bench(run_dir: str | Path, questions: list[str], device: str = "auto", repeats: int = 1) -> dict:
    """Added latency of the probe = separate prompt pass + probe, vs plain generation alone.
    (A production server could reuse the prompt pass's KV cache; this measures the simple version.)"""
    svc = ConfidenceService(run_dir, device)
    plain, added = [], []
    for _ in range(repeats):
        for q in questions:
            prompt = render(svc.tok, answer_messages(q))
            g = generate(svc.model, svc.tok, [prompt], svc.max_new_tokens, capture=False)
            plain.append(g.seconds * 1000)
            _, ms = svc.risk_before_answer(prompt)
            added.append(ms)
    out = {"device": svc.device, "n": len(plain),
           "generate_ms_p50": float(np.percentile(plain, 50)), "generate_ms_p95": float(np.percentile(plain, 95)),
           "added_ms_p50": float(np.percentile(added, 50)), "added_ms_p95": float(np.percentile(added, 95))}
    (Path(run_dir) / "bench.json").write_text(json.dumps(out, indent=2))
    return out


def make_app(run_dir: str | Path, device: str = "auto"):
    from fastapi import FastAPI
    from pydantic import BaseModel

    svc = ConfidenceService(run_dir, device)
    app = FastAPI(title="kwik confidence API")

    class Q(BaseModel):
        question: str

    @app.post("/answer")
    def answer(q: Q):
        return svc.answer(q.question)

    @app.get("/health")
    def health():
        return {"ok": True, "layer": svc.layer, "device": svc.device}

    return app
