"""Model loading and the one instrumented generation call everything else uses."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(dtype: str, device: str):
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]


def load_model(model_id: str, device: str = "auto", dtype: str = "auto", load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # so position -1 is always the last real prompt token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers
    from packaging.version import Version

    # transformers renamed torch_dtype -> dtype in 4.56
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {dtype_key: pick_dtype(dtype, device)}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
                  "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return model, tok, device


def layer_indices(num_hidden_states: int, stride: int) -> list[int]:
    """hidden_states has num_layers + 1 entries (index 0 = embeddings). Keep every `stride`-th
    transformer layer plus the last one."""
    idx = list(range(1, num_hidden_states, stride))
    if idx[-1] != num_hidden_states - 1:
        idx.append(num_hidden_states - 1)
    return idx


@dataclass
class GenOutput:
    texts: list[str]               # raw decoded answers
    pre: np.ndarray | None         # (B, L, D) state at the last prompt token: before any answer token
    end: np.ndarray | None         # (B, L, D) state when the model decides to stop (reads its last answer token)
    logprob_mean: np.ndarray       # (B,) mean log-prob of the answer tokens
    logprob_min: np.ndarray        # (B,) lowest token log-prob in the answer
    n_tokens: np.ndarray           # (B,) answer length in tokens
    seconds: float                 # wall time for the whole batch
    layers: list[int]


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, capture: bool = True,
             layer_stride: int = 2, do_sample: bool = False, temperature: float = 1.0,
             num_return_sequences: int = 1) -> GenOutput:
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    t0 = time.perf_counter()
    gen_kwargs = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True,
                      output_hidden_states=capture, pad_token_id=tok.pad_token_id,
                      num_return_sequences=num_return_sequences)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(**enc, **gen_kwargs)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    prompt_len = enc["input_ids"].shape[1]
    new_tokens = out.sequences[:, prompt_len:]                       # (B*, S)
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    gen_cfg_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_cfg_eos, int):
        stop_ids.add(gen_cfg_eos)
    elif isinstance(gen_cfg_eos, (list, tuple)):
        stop_ids.update(gen_cfg_eos)
    stop_ids.discard(None)

    B, S = new_tokens.shape
    lengths = np.zeros(B, dtype=int)
    for b in range(B):
        n = 0
        for t in new_tokens[b].tolist():
            if t in stop_ids:
                break
            n += 1
        lengths[b] = n

    # token log-probs of the chosen answer tokens
    logprobs = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)  # (B*, S, V)
    chosen = logprobs.gather(-1, new_tokens[:, : logprobs.shape[1]].unsqueeze(-1)).squeeze(-1).cpu().numpy()
    lp_mean = np.array([chosen[b, : max(lengths[b], 1)].mean() for b in range(B)])
    lp_min = np.array([chosen[b, : max(lengths[b], 1)].min() for b in range(B)])

    texts = tok.batch_decode([new_tokens[b, : lengths[b]] for b in range(B)], skip_special_tokens=True)

    pre = end = None
    layers: list[int] = []
    if capture:
        steps = out.hidden_states                     # tuple over steps; each a tuple over layers
        layers = layer_indices(len(steps[0]), layer_stride)
        pre = torch.stack([steps[0][l][:, -1, :] for l in layers], dim=1).float().cpu().numpy()
        end_rows = []
        for b in range(B):
            # step k's forward pass reads token k-1; the step that reads the last answer token is k = length
            k = int(min(max(lengths[b], 1), len(steps) - 1))
            end_rows.append(torch.stack([steps[k][l][b, -1, :] for l in layers], dim=0))
        end = torch.stack(end_rows, dim=0).float().cpu().numpy()
    return GenOutput(texts, pre, end, lp_mean, lp_min, lengths, seconds, layers)
