"""Step 5: the confidence API and its latency benchmark.

POST /answer {"question": "..."} ->
  {"answer", "risk_wrong", "risk_from": "before answer, layer N", "timings_ms": {...}}

The risk score comes from the "pre" probe. A forward hook on the probe's layer reads the last prompt
token's hidden state during generation's own first (prefill) pass, so the score exists before the
first answer token is produced and costs no extra model pass. (Version 1 ran a separate prompt pass,
which the benchmark showed added ~65% to latency on CPU; see README "What broke".)

`risk_without_generating` is for gating: one prefill pass, no answer, e.g. to decide whether to look
something up before answering at all.
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


def _decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("unsupported architecture: cannot find decoder layers")


def _final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("unsupported architecture: cannot find final norm")


class ConfidenceService:
    def __init__(self, run_dir: str | Path, device: str = "auto", max_new_tokens: int = 24, model=None, tok=None):
        run_dir = Path(run_dir)
        bundle = joblib.load(run_dir / "probe_pre.joblib")
        self.probe, self.layer = bundle["model"], bundle["layer"]
        if model is None:
            self.model, self.tok, self.device = load_model(bundle["model_id"], device)
        else:
            self.model, self.tok, self.device = model, tok, str(model.device)
        self.max_new_tokens = max_new_tokens
        layers = _decoder_layers(self.model)
        # hidden_states[l] is the output of decoder layer l-1, except the last entry, which HF
        # returns after the final norm
        self._module = _final_norm(self.model) if self.layer == len(layers) else layers[self.layer - 1]

    def _capture(self):
        store: dict = {}

        def hook(_module, _inputs, output):
            if "h" not in store:  # first call = the prefill pass over the whole prompt
                h = output[0] if isinstance(output, tuple) else output
                store["h"] = h[0, -1, :].detach().float().cpu().numpy().reshape(1, -1)

        return store, self._module.register_forward_hook(hook)

    def _score(self, h: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        risk = float(self.probe.predict_proba(h)[0, 1])
        return risk, (time.perf_counter() - t0) * 1000

    @torch.no_grad()
    def risk_without_generating(self, prompt: str) -> float:
        store, handle = self._capture()
        try:
            enc = self.tok([prompt], return_tensors="pt").to(self.model.device)
            self.model(**enc)
        finally:
            handle.remove()
        return self._score(store["h"])[0]

    def answer(self, question: str) -> dict:
        prompt = render(self.tok, answer_messages(question))
        store, handle = self._capture()
        try:
            g = generate(self.model, self.tok, [prompt], self.max_new_tokens, capture=False)
        finally:
            handle.remove()
        risk, probe_ms = self._score(store["h"])
        return {"answer": clean_prediction(g.texts[0]), "risk_wrong": round(risk, 4),
                "risk_from": f"before answer, layer {self.layer}",
                "timings_ms": {"generate_with_hook": round(g.seconds * 1000, 1), "probe": round(probe_ms, 2)}}


def bench(run_dir: str | Path, questions: list[str], device: str = "auto", repeats: int = 1) -> dict:
    """Added latency = (generation with the hook + probe) minus plain generation, measured in
    alternating pairs on the same question so machine noise affects both sides."""
    svc = ConfidenceService(run_dir, device)
    warm = render(svc.tok, answer_messages(questions[0]))  # first calls pay one-off setup costs
    generate(svc.model, svc.tok, [warm], svc.max_new_tokens, capture=False)
    svc.answer(questions[0])
    plain, hooked, probe_ms = [], [], []
    for _ in range(repeats):
        for i, q in enumerate(questions):
            prompt = render(svc.tok, answer_messages(q))
            order = (0, 1) if i % 2 == 0 else (1, 0)  # alternate which side runs first
            for side in order:
                if side == 0:
                    g = generate(svc.model, svc.tok, [prompt], svc.max_new_tokens, capture=False)
                    plain.append(g.seconds * 1000)
                else:
                    t0 = time.perf_counter()
                    out = svc.answer(q)
                    hooked.append((time.perf_counter() - t0) * 1000)
                    probe_ms.append(out["timings_ms"]["probe"])
    plain, hooked = np.array(plain), np.array(hooked)
    added = hooked - plain
    out = {"device": svc.device, "n": int(len(plain)),
           "generate_ms_p50": float(np.percentile(plain, 50)), "generate_ms_p95": float(np.percentile(plain, 95)),
           "with_risk_ms_p50": float(np.percentile(hooked, 50)), "with_risk_ms_p95": float(np.percentile(hooked, 95)),
           "added_ms_p50": float(np.percentile(added, 50)), "added_ms_p95": float(np.percentile(added, 95)),
           "probe_only_ms_p50": float(np.percentile(probe_ms, 50))}
    out["added_pct_of_generate_p50"] = 100 * out["added_ms_p50"] / out["generate_ms_p50"]
    out["examples"] = [{"question": q, **svc.answer(q)} for q in questions[:5]]
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
