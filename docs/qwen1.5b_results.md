# Results: qwen1.5b

Model: `Qwen/Qwen2.5-1.5B-Instruct`. Questions: {'train': 1320, 'val': 445, 'test': 447, 'transfer': 800}. Error rate by split: {'train': 0.769, 'val': 0.728, 'test': 0.747, 'transfer': 0.873}.

Higher AUROC is better (spotting wrong answers). Lower ECE, Brier and AURC are better.
acc@80% = accuracy if the system answers only the 80% of questions it is least worried about.

## Test split (same datasets as training)

| method | n | auroc [95% CI] | ece | brier | aurc | acc@80% |
|---|---|---|---|---|---|---|
| probe (after answer, layer 19) | 447 | 0.872 [0.83, 0.91] | 0.071 | 0.126 | 0.488 | 0.313 |
| logprob (min) | 447 | 0.789 [0.74, 0.83] | 0.048 | 0.152 | 0.561 | 0.307 |
| probe (before answer, layer 19) | 447 | 0.777 [0.72, 0.82] | 0.123 | 0.174 | 0.558 | 0.302 |
| logprob (mean) | 447 | 0.773 [0.72, 0.82] | 0.070 | 0.157 | 0.566 | 0.299 |
| stated confidence | 447 | 0.763 [0.71, 0.81] | 0.120 | 0.164 | 0.545 | 0.293 |
| sampling agreement | 447 | 0.744 [0.69, 0.79] | 0.056 | 0.165 | 0.575 | 0.310 |

## Transfer split (dataset never seen in training)

| method | n | auroc [95% CI] | ece | brier | aurc | acc@80% |
|---|---|---|---|---|---|---|
| logprob (min) | 800 | 0.874 [0.84, 0.90] | 0.051 | 0.088 | 0.704 | 0.158 |
| probe (before answer, layer 19) | 800 | 0.862 [0.83, 0.90] | 0.094 | 0.101 | 0.703 | 0.159 |
| logprob (mean) | 800 | 0.862 [0.82, 0.90] | 0.038 | 0.084 | 0.700 | 0.156 |
| sampling agreement | 800 | 0.840 [0.80, 0.88] | 0.088 | 0.094 | 0.728 | 0.155 |
| probe (after answer, layer 19) | 800 | 0.829 [0.78, 0.87] | 0.070 | 0.096 | 0.710 | 0.152 |
| stated confidence | 800 | 0.735 [0.68, 0.79] | 0.163 | 0.129 | 0.755 | 0.147 |

Correct answers available for evaluation: test 113, transfer 102. Fewer than ~100 makes the intervals wide.

## Is the before-answer probe better? (paired bootstrap, same questions)

| split | versus | AUROC difference [95% CI] | verdict |
|---|---|---|---|
| test | probe (after answer, layer 19) | -0.096 [-0.142, -0.052] | probe worse |
| test | logprob (mean) | +0.003 [-0.062, +0.067] | no clear difference |
| test | logprob (min) | -0.013 [-0.069, +0.045] | no clear difference |
| test | stated confidence | +0.014 [-0.050, +0.073] | no clear difference |
| test | sampling agreement | +0.032 [-0.032, +0.093] | no clear difference |
| transfer | probe (after answer, layer 19) | +0.034 [+0.001, +0.067] | probe better |
| transfer | logprob (mean) | +0.000 [-0.033, +0.035] | no clear difference |
| transfer | logprob (min) | -0.012 [-0.042, +0.018] | no clear difference |
| transfer | stated confidence | +0.127 [+0.081, +0.178] | probe better |
| transfer | sampling agreement | +0.023 [-0.019, +0.066] | no clear difference |

## Cost

- Answer generation: 0.997 s per question (batched, includes saving hidden states)
- Stated confidence: +1.116 s per question (extra call)
- Sampling agreement: +2.583 s per question (extra samples)

## Chosen probes (picked on validation only)

- pre: layer 19, C=0.01, val AUROC 0.820
- end: layer 19, C=0.01, val AUROC 0.893

Stated-confidence replies that contained a usable number: 1.0

![layer sweep](qwen1.5b_layer_sweep.png)


Automatic correctness labels must be checked against hand labels (`kwik handcheck-agree`) before these numbers are quoted.
