# Does It Know What It Knows?

Small probes read an open language model's hidden states and predict **"this answer will be wrong"**,
in one case before the model writes its first answer token. The question is whether that internal
signal is a cheaper and better warning than the usual alternatives: asking the model how sure it is,
or sampling it several times.

This is metacognition measured, not claimed: one of the indicator properties in
[Consciousness in Artificial Intelligence](https://arxiv.org/abs/2308.08708) (Butlin, Long et al., 2023),
tested as a working, benchmarked service.

## Results

**Qwen2.5 1.5B (laptop CPU) and 7B (free Colab T4, 4-bit), the same 3,000 closed-book questions.**
AUROC for spotting a wrong answer, with 95% bootstrap intervals. "Test" is held-out questions from
TriviaQA and NQ-Open, the datasets the probes were trained on (n=447). "Transfer" is PopQA, a dataset
the probes never saw (n=800).

| Method | 1.5B test | 1.5B transfer | 7B test | 7B transfer | Extra time per question |
|---|---|---|---|---|---|
| Probe, after answer | **0.872** [0.83, 0.91] | 0.829 [0.78, 0.87] | **0.863** [0.83, 0.90] | 0.858 [0.82, 0.90] | not served |
| Probe, before answer | 0.777 [0.72, 0.82] | 0.862 [0.83, 0.90] | 0.783 [0.74, 0.83] | 0.844 [0.80, 0.89] | +18 ms (1.5B, p50) |
| Token log-prob, min | 0.789 [0.74, 0.83] | **0.874** [0.84, 0.90] | 0.813 [0.77, 0.85] | **0.895** [0.86, 0.92] | 0 |
| Token log-prob, mean | 0.773 [0.72, 0.82] | 0.862 [0.82, 0.90] | 0.806 [0.77, 0.85] | **0.895** [0.87, 0.92] | 0 |
| Sampling agreement (k=3) | 0.744 [0.69, 0.79] | 0.840 [0.80, 0.88] | 0.799 [0.76, 0.84] | 0.839 [0.80, 0.87] | +2.6 s (1.5B), +1.3 s (7B) |
| Stated confidence | 0.763 [0.71, 0.81] | 0.735 [0.68, 0.79] | 0.708 [0.67, 0.75] | 0.725 [0.68, 0.76] | +1.1 s (1.5B), +0.6 s (7B) |

Error rates: 1.5B 75% test / 87% transfer; 7B 60% / 86%. Correct answers available for evaluation:
1.5B 113 / 102; 7B 179 / 116.

**Findings** (paired bootstrap on the same questions):

1. **On familiar question types, the after-answer probe beats every alternative at both sizes.**
   Against the best token-probability score: +0.08 [+0.04, +0.13] at 1.5B, +0.05 [+0.01, +0.09] at 7B.
   Against sampling: +0.13 and +0.06. Against stated confidence: +0.11 and +0.15. The margin shrinks
   as the model's own token probabilities get better.
2. **The advantage does not transfer, at either size.** On PopQA the after-answer probe falls behind
   the free token-probability score: −0.045 [−0.085, −0.005] at 1.5B, −0.037 [−0.072, −0.002] at 7B.
   The probes partly learn cues specific to their training datasets.
3. **The before-answer probe sees most of the signal before a single answer token exists** (0.78
   test, 0.84–0.86 transfer), early enough to fetch sources instead of guessing. At 1.5B it transferred
   better than the after-answer probe (+0.034 [+0.001, +0.067]); at 7B it did not (−0.014 [−0.048,
   +0.016]), so that advantage is not established.
4. **Asking the model how sure it is is the weakest signal at both sizes**, and sampling costs extra
   time without beating free token probabilities.

**Serving cost** (`kwik bench`, 1.5B, 50 questions, Core Ultra laptop CPU): the API reads the probe's
layer during generation's own first pass, so the risk score exists before the first answer token. It
adds 18 ms at the median (0.8% of a 2.2 s answer); the probe itself takes 1.5 ms. The 95th percentile
(188 ms) is mostly machine noise. Details: `docs/qwen1.5b_bench.json`.

**Labels:** automatic correctness scoring agreed with 200 hand labels on 97.5% of answers (kappa 0.92;
78 from the 0.5B pilot, 122 from the 1.5B run). Of the 5 disagreements, 3 were the scorer being too
strict (for example "Catholicism" against "Catholic Church"), and an early one led to a stricter
rule for hedged and list answers.

| 1.5B | 7B |
|---|---|
| ![1.5B layer sweep](docs/qwen1.5b_layer_sweep.png) | ![7B layer sweep](docs/qwen7b_layer_sweep.png) |

At both sizes the signal rises through the middle of the network and peaks around three quarters of
the way up (layer 19 of 28 at 1.5B, layer 21 of 28 at 7B), then falls slightly near the output.
Full reports: `docs/qwen1.5b_results.md`, `docs/qwen7b_results.md`.

## How it works

```
questions (TriviaQA, NQ-Open; PopQA held out)
   │
   ▼
kwik collect ── open model answers each question (greedy)
   │              saves: answer, correct/incorrect, token log-probs,
   │              hidden states at two positions × every 2nd layer
   │                "pre": last prompt token (before any answer token)
   │                "end": when the model reads its own last answer token
   ▼
kwik baselines ─ stated confidence (extra call) + sampling agreement (k extra samples)
   ▼
kwik train ───── logistic-regression probe per layer × position; layer, position and
   │              regularisation chosen on validation only; test and transfer used once
   ▼
kwik report ──── results.md, layer curve, risk-coverage plot
   ▼
kwik serve ───── FastAPI: POST /answer → answer + risk score (from "pre") + timings
kwik bench ───── added latency of the probe vs plain generation
```

**Design decisions**

- **Label:** wrong = neither exact match nor whole-word containment of any gold alias (predictions over
  10 words get no containment credit). Checked against 200 hand labels with `kwik handcheck-*`
  before any number is quoted.
- **Splits:** TriviaQA + NQ-Open pooled and split 60/20/20, stratified by dataset. PopQA (long-tail
  entities) is never seen in training, which tests whether the probe transfers.
- **Fair calibration:** every baseline gets a Platt mapping fitted on validation, so ECE compares
  probabilities with probabilities. AUROC is unaffected by that mapping.
- **Honest naming:** "sampling agreement" is a cheap stand-in for semantic entropy (it matches answers by
  normalized text rather than clustering them by meaning).
- **Resumable:** `collect` writes one shard per batch and skips finished questions when re-run.

## Run it

### 1. Pilot on your laptop (CPU, about 30–90 min; time it)

Anaconda Prompt:

```
conda create -n kwik python=3.11 -y
conda activate kwik
git clone https://github.com/feliciachen01/kwik.git
cd kwik
pip install -e ".[serve,dev]"
pytest -q
kwik all --config configs/pilot.yaml
```

Then open `runs/pilot-qwen0.5b/results.md`. Run long jobs overnight, never during the Truthful
Companion's latency tests.

### 2. Check the labels by hand

```
kwik handcheck-export --config configs/pilot.yaml
```

Fill the `human_correct` column of `runs/pilot-qwen0.5b/handcheck.csv` with 1 or 0, then:

```
kwik handcheck-agree --config configs/pilot.yaml
```

### 3. Bigger runs

- `configs/laptop-1.5b.yaml`: Qwen2.5-1.5B on CPU, 3,000 questions (about 2 h 45 min on a Core Ultra laptop). Full report: `docs/qwen1.5b_results.md`.
- `notebooks/colab_7b.ipynb`: Qwen2.5-7B in 4-bit on a free Colab T4, same 3,000 questions (about 1 h 30 min). Full report: `docs/qwen7b_results.md`.

### 4. Serve and benchmark

```
kwik bench --config configs/laptop-1.5b.yaml
kwik serve --config configs/laptop-1.5b.yaml
```

```
curl -X POST localhost:8000/answer -H "Content-Type: application/json" -d "{\"question\": \"Who wrote Hamlet?\"}"
```

## Data and models

| Source | Hub ID | Use | License |
|---|---|---|---|
| TriviaQA | `mandarjoshi/trivia_qa` (rc.nocontext, validation) | train / val / test | see dataset card |
| NQ-Open | `google-research-datasets/nq_open` (validation) | train / val / test | CC BY-SA 3.0 |
| PopQA | `akariasai/PopQA` (test) | transfer only | see dataset card |
| Qwen2.5 Instruct 0.5B / 1.5B / 7B | `Qwen/Qwen2.5-*-Instruct` | models | Apache 2.0 |

## Prior work

- Kossen et al., [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) (2024): probes on hidden
  states approximate semantic entropy cheaply. This repo adds a "before the answer" position,
  transfer to a held-out long-tail dataset, and a served API with measured overhead.

## What broke

| What happened | Why | What changed |
|---|---|---|
| The 0.5B pilot's intervals were too wide to separate any methods | The model got ~10% of answers right, leaving ~30 correct answers per split | Moved to 1.5B (~100+ correct per split) and added bootstrap intervals and paired comparisons to every report |
| "New York City, Tokyo, and Beijing" was scored correct for "largest cities" | Containment credit let one right item inside a wrong list count | Hedges and lists now need every part to be an accepted answer; found by the hand check (77/78 agreement) |
| Re-scoring the pilot flipped a label that should not have changed | Only the first 20 accepted answers per question were saved; TriviaQA has more aliases than that | All accepted answers are saved now |
| The in-distribution win did not hold on PopQA | The probes partly learn cues specific to TriviaQA and NQ | Reported as a finding; it held at both 1.5B and 7B, so it is not a small-model artefact |
| The first API benchmark added 1.3 s per question (+65%) | The API ran a separate full pass over the prompt just to read one layer | A forward hook now reads the layer during generation's own first pass: +18 ms (0.8%), same risk scores |
| Models could not be downloaded in sandboxed environments | Hugging Face was blocked by the network allowlist | Runs happen on a normal Windows install (one-click `run_*.bat` scripts) and on Colab |

## Tests

`pytest -q` runs scoring and metric unit tests plus an offline end-to-end run on a tiny randomly
initialised model. That run checks the plumbing only; its numbers mean nothing.
