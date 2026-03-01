from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match_norm(pred: str, target: str) -> float:
    return 1.0 if _norm(pred) == _norm(target) else 0.0


def _tokens(s: str) -> list[str]:
    return _norm(s).split()


def token_f1(pred: str, target: str) -> float:
    p = _tokens(pred)
    t = _tokens(target)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    t_count: dict[str, int] = {}
    for tok in t:
        t_count[tok] = t_count.get(tok, 0) + 1
    common = 0
    for tok in p:
        if t_count.get(tok, 0) > 0:
            common += 1
            t_count[tok] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(t)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[m][n]


def rouge_l_f1(pred: str, target: str) -> float:
    p = _tokens(pred)
    t = _tokens(target)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    lcs = _lcs_len(p, t)
    prec = lcs / len(p)
    rec = lcs / len(t)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def metric_name_from_sample(sample: dict[str, Any]) -> str:
    return str(sample.get("metadata", {}).get("metric", "token_f1"))


def score_sample(pred: str, target: str, metric: str) -> float:
    if metric == "exact_match_norm":
        return exact_match_norm(pred, target)
    if metric == "rougeL":
        return rouge_l_f1(pred, target)
    return token_f1(pred, target)
