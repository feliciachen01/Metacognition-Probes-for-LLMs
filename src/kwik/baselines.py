"""Step 2: the alternatives the probe has to beat.

  logprob     free: mean token log-prob of the answer (already saved by `collect`)
  stated      ask the model for a 0-100 probability that its own answer is correct (one extra call)
  sampling    draw K extra answers at temperature 1.0; risk = share that disagree with the greedy answer
              (a cheap stand-in for semantic entropy, which also clusters answers by meaning)

Writes runs/<run_name>/baselines.jsonl with the score and the seconds each method cost per question.
"""

from __future__ import annotations

import json
import re

from .config import Config
from .model import generate, load_model
from .prompts import answer_messages, confidence_messages, render
from .scoring import clean_prediction, same_answer

_INT = re.compile(r"\d{1,3}")


def parse_confidence(text: str) -> float | None:
    m = _INT.search(text)
    if not m:
        return None
    return min(int(m.group()), 100) / 100.0


def run_baselines(cfg: Config, model=None, tok=None, splits=("test", "transfer", "val")) -> None:
    run_dir = cfg.run_dir
    answers = [json.loads(l) for l in (run_dir / "answers.jsonl").read_text().splitlines() if l.strip()]
    out_path = run_dir / "baselines.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["uid"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [a for a in answers if a["split"] in splits and a["uid"] not in done]
    print(f"[baselines] {len(todo)} questions to score")
    if not todo:
        return
    if model is None:
        model, tok, _ = load_model(cfg.model_id, cfg.device, cfg.dtype, cfg.load_in_4bit)

    with open(out_path, "a") as fout:
        for start in range(0, len(todo), cfg.batch_size):
            batch = todo[start : start + cfg.batch_size]
            # stated confidence
            conf_prompts = [render(tok, confidence_messages(a["question"], a["prediction"])) for a in batch]
            c = generate(model, tok, conf_prompts, max_new_tokens=6, capture=False)
            # sampling agreement
            ans_prompts = [render(tok, answer_messages(a["question"])) for a in batch]
            s = generate(model, tok, ans_prompts, cfg.max_new_tokens, capture=False, do_sample=True,
                         temperature=cfg.sampling_temperature, num_return_sequences=cfg.sampling_k)
            for i, a in enumerate(batch):
                samples = [clean_prediction(t) for t in s.texts[i * cfg.sampling_k : (i + 1) * cfg.sampling_k]]
                agree = sum(same_answer(a["prediction"], x) for x in samples) / cfg.sampling_k
                fout.write(json.dumps({
                    "uid": a["uid"],
                    "stated_conf": parse_confidence(c.texts[i]), "stated_raw": c.texts[i],
                    "sampling_agree": agree, "samples": samples,
                    "stated_seconds": c.seconds / len(batch), "sampling_seconds": s.seconds / len(batch),
                }) + "\n")
            fout.flush()
            print(f"[baselines] {start + len(batch)}/{len(todo)}")
