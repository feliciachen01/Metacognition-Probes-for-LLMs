"""Automatic correctness scoring for short answers.

The label is "correct" if the normalized prediction equals a normalized gold alias (exact match),
or if a gold alias appears as whole words inside a short prediction (contains match).
Automatic labels are checked against 200 hand labels (`kwik handcheck`) before anything is reported.
"""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”"}

MAX_CONTAINS_WORDS = 10  # long rambling predictions don't get credit for containing the answer


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def clean_prediction(raw: str) -> str:
    """First non-empty line, without a leading 'Answer:' and trailing period."""
    for line in raw.strip().splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", line, flags=re.I)
            return line.rstrip(" .")
    return ""


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return bool(p) and any(p == normalize(g) for g in golds)


def contains_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    if not p or len(p.split()) > MAX_CONTAINS_WORDS:
        return False
    padded = f" {p} "
    return any(normalize(g) and f" {normalize(g)} " in padded for g in golds)


def score(pred: str, golds: list[str]) -> dict:
    em = exact_match(pred, golds)
    cm = contains_match(pred, golds)
    return {"em": em, "contains": cm, "correct": em or cm}


def same_answer(a: str, b: str) -> bool:
    """Used by the sampling baseline: do two predictions say the same thing?"""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na == nb or f" {na} " in f" {nb} " or f" {nb} " in f" {na} "
