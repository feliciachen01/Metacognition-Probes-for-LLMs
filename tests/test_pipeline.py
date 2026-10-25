"""End-to-end plumbing test on a tiny randomly initialised model, fully offline.

It proves every step runs and files line up. It says nothing about real probe quality: a random model
gets every answer wrong, so the test overwrites correctness labels at random to exercise training.
"""

import json
import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kwik.config import Config, DatasetSpec  # noqa: E402

WORDS = ["paris", "london", "rome", "who", "what", "is", "the", "capital", "of", "france", "italy", "england",
         "answer", "question", "a", "b", "c", "1", "2", "3", "system", "user", "assistant", ":", "?", "."]


def make_tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=1, pad_token_id=0, bos_token_id=1, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("kwik")
    model_dir = root / "tiny"
    make_tiny_model(model_dir)
    qa = root / "qa.jsonl"
    rng = random.Random(0)
    lines = [json.dumps({"uid": f"q{i}", "dataset": "toyA" if i < 80 else "toyB",
                         "question": f"what is the capital of {rng.choice(['france', 'italy', 'england'])} ?",
                         "answers": [rng.choice(["paris", "rome", "london"])]}) for i in range(110)]
    qa.write_text("\n".join(lines[:80]))
    qa_transfer = root / "qa_transfer.jsonl"
    qa_transfer.write_text("\n".join(lines[80:]))
    cfg = Config(run_name="t", model_id=str(model_dir), out_dir=str(root / "runs"),
                 datasets=[DatasetSpec(name="local", n=80, role="train", path=str(qa)),
                           DatasetSpec(name="local", n=30, role="transfer", path=str(qa_transfer))],
                 batch_size=8, max_new_tokens=5, layer_stride=2, device="cpu", sampling_k=3)
    return cfg, qa


def test_end_to_end(run):
    cfg, qa = run
    from kwik.baselines import run_baselines
    from kwik.collect import collect, load_features
    from kwik.handcheck import export
    from kwik.probes import train_and_evaluate
    from kwik.report import write_report
    from kwik.service import ConfidenceService, bench

    collect(cfg)
    qs = [json.loads(l) for l in (cfg.run_dir / "questions.jsonl").read_text().splitlines()]
    splits = {s: sum(q["split"] == s for q in qs) for s in ("train", "val", "test", "transfer")}
    assert splits == {"train": 48, "val": 16, "test": 16, "transfer": 30}

    uids, feats, layers = load_features(cfg.run_dir)
    assert feats["pre"].shape[0] == 110 and feats["pre"].shape == feats["end"].shape
    assert layers == [1, 3, 4]  # 4 layers, stride 2, plus the last
    assert feats["pre"].shape[2] == 32

    # resume: a second call does nothing new
    n_shards = len(list((cfg.run_dir / "shards").glob("*.npz")))
    collect(cfg)
    assert len(list((cfg.run_dir / "shards").glob("*.npz"))) == n_shards

    # test-only: random labels so the probe has two classes to learn from
    ans_path = cfg.run_dir / "answers.jsonl"
    rng = random.Random(1)
    rows = [json.loads(l) for l in ans_path.read_text().splitlines()]
    for r in rows:
        r["correct"] = rng.random() < 0.5
    ans_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    run_baselines(cfg)
    base = [json.loads(l) for l in (cfg.run_dir / "baselines.jsonl").read_text().splitlines()]
    assert len(base) == sum(r["split"] in ("val", "test", "transfer") for r in rows)
    assert all(0 <= b["sampling_agree"] <= 1 for b in base)

    summary = train_and_evaluate(cfg.run_dir)
    assert set(summary["choices"]) == {"pre", "end"}
    assert (cfg.run_dir / "probe_pre.joblib").exists()
    import pandas as pd

    res = pd.read_csv(cfg.run_dir / "results.csv")
    assert set(res["split"]) == {"test", "transfer"}
    methods = set(res["method"])
    assert {"logprob (mean)", "sampling agreement"} <= methods
    # the random model never writes a number when asked for confidence, so that baseline is skipped
    # rather than silently filled; real models parse fine (the parse rate is reported)
    assert summary["stated_parse_rate"] == 0.0 and "stated confidence" not in methods
    assert sum(m.startswith("probe") for m in methods) == 2

    svc = ConfidenceService(cfg.run_dir, device="cpu", max_new_tokens=5)
    out = svc.answer("what is the capital of france ?")
    assert 0.0 <= out["risk_wrong"] <= 1.0 and "generate" in out["timings_ms"]

    b = bench(cfg.run_dir, ["what is the capital of italy ?"] * 3, device="cpu")
    assert b["n"] == 3 and b["added_ms_p50"] > 0

    report = write_report(cfg.run_dir)
    assert "Test split" in report.read_text()
    assert (cfg.run_dir / "risk_coverage.png").exists()
    assert export(cfg.run_dir, n=20).exists()
