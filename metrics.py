"""
coeff_pred_metric.py
====================
Evaluation metrics for the Taylor coefficient prediction task.

Metrics computed:
  1. token_accuracy      — fraction of non-PAD tokens predicted correctly
  2. sentence_accuracy   — fraction of sequences where every non-PAD token matches
  3. correct_expression  — fraction of sequences that form a valid prefix expression
                           (i.e. can be converted to infix without a parse error)
  4. split_segments      — split a token-ID list on <BREAK> into per-coefficient
                           segments (used for per-coefficient evaluation)
  5. per_segment_metrics — compute metrics for each coefficient segment individually

All tensor functions accept raw (padded) batch tensors and a pad_id, matching
the shape produced by CoeffPredDataset.collate_fn.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from dataset import BREAK_ID, EOS_ID, N_COEFFS, PAD_ID, VOCAB


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TOKEN-LEVEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def token_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids:  torch.Tensor,
    pad_id:   int,
) -> float:
    """Fraction of non-PAD target tokens that are predicted correctly.

    Parameters
    ----------
    pred_ids : (B, L)  — predicted token IDs (greedy argmax of logits)
    tgt_ids  : (B, L)  — ground-truth token IDs (padded)
    pad_id   : int     — padding token ID to exclude from the count

    Returns
    -------
    float in [0, 1]
    """
    mask    = tgt_ids != pad_id                  # (B, L)  True for real tokens
    correct = (pred_ids == tgt_ids) & mask       # (B, L)
    total   = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SENTENCE-LEVEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def sentence_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids:  torch.Tensor,
    pad_id:   int,
) -> float:
    """Fraction of sequences where every non-PAD position matches exactly.

    PAD positions are masked to pad_id in both tensors before comparison so
    trailing padding never counts as a correct match.

    Parameters
    ----------
    pred_ids : (B, L)  — predicted token IDs
    tgt_ids  : (B, L)  — ground-truth token IDs (padded)
    pad_id   : int     — padding token ID

    Returns
    -------
    float in [0, 1]
    """
    mask        = tgt_ids != pad_id                          # (B, L)
    pred_masked = pred_ids.masked_fill(~mask, pad_id)        # (B, L)
    tgt_masked  = tgt_ids.masked_fill(~mask, pad_id)        # (B, L)
    match       = (pred_masked == tgt_masked).all(dim=1)     # (B,)
    return match.float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CORRECT EXPRESSION (prefix-validity check)
# ══════════════════════════════════════════════════════════════════════════════

# Operator / symbol sets that mirror the dataset vocabulary
_BINARY_OPS = frozenset({"+", "-", "*", "/", "**"})
_UNARY_OPS  = frozenset({
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh",
})
_LEAF_SYMS  = frozenset({"x", "a", "pi", "E", "oo", "-oo"})
_DIGITS     = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
_SIGN_TOKS  = frozenset({"+", "-"})
_SPECIAL    = frozenset({"<PAD>", "<SOS>", "<EOS>", "<UNK>"})


def _parse_one(tokens: List[str], pos: int) -> int:
    """Consume exactly one prefix expression from *tokens* starting at *pos*.

    Numbers are represented as a sign token ('+' or '-') followed by one or
    more digit tokens — consecutive digits are combined into a single integer,
    matching the encoding used in dataset-generation.py.

    Returns the position immediately after the consumed expression.
    Raises ``ValueError`` if the stream is malformed or exhausted.
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of token stream")

    tok = tokens[pos]

    # ── Number: sign token followed by one-or-more digit tokens ──────────────
    # Digits are individual vocab tokens ('0'..'9'); combine them into one int.
    if tok in _SIGN_TOKS and pos + 1 < len(tokens) and tokens[pos + 1] in _DIGITS:
        pos += 2  # skip sign and first digit
        while pos < len(tokens) and tokens[pos] in _DIGITS:
            pos += 1   # consume additional digit tokens (multi-digit numbers)
        return pos

    # ── Leaf symbols ──────────────────────────────────────────────────────────
    if tok in _LEAF_SYMS:
        return pos + 1

    # ── Binary operators ──────────────────────────────────────────────────────
    if tok in _BINARY_OPS:
        pos = _parse_one(tokens, pos + 1)   # left operand
        pos = _parse_one(tokens, pos)        # right operand
        return pos

    # ── Unary functions ───────────────────────────────────────────────────────
    if tok in _UNARY_OPS:
        return _parse_one(tokens, pos + 1)

    raise ValueError(f"Unexpected token {tok!r} at position {pos}")


def _is_valid_prefix(tokens: List[str]) -> bool:
    """Return True iff *tokens* is a syntactically complete prefix expression.

    A complete expression is one where the parser consumes ALL tokens exactly.
    An empty token list is considered invalid.
    """
    if not tokens:
        return False
    try:
        end = _parse_one(tokens, 0)
        return end == len(tokens)
    except ValueError:
        return False


def correct_expression(
    pred_ids: torch.Tensor,
    pad_id:   int,
) -> float:
    """Fraction of predicted sequences that form a valid prefix expression.

    For each sequence the tokens are collected up to (but not including) the
    first EOS or PAD token; consecutive digit tokens are treated as a single
    integer (matching dataset-generation.py encoding).

    Parameters
    ----------
    pred_ids : (B, L)  — predicted token IDs
    pad_id   : int     — padding token ID (sequence truncated here)

    Returns
    -------
    float in [0, 1]
    """
    B = pred_ids.shape[0]
    n_valid = 0
    for i in range(B):
        toks: List[str] = []
        for tid in pred_ids[i].tolist():
            if tid == EOS_ID or tid == pad_id:
                break
            token = VOCAB[tid] if 0 <= tid < len(VOCAB) else None
            if token is None or token in _SPECIAL:
                break
            toks.append(token)
        if _is_valid_prefix(toks):
            n_valid += 1
    return n_valid / B if B > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMPUTE ALL METRICS FROM LOGITS  (convenience wrapper used in training loop)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    logits:  torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id:  int,
) -> dict:
    """Compute token accuracy, sentence accuracy, and correct_expression from logits.

    The logits correspond to predictions for tgt[:, 1:] (teacher-forced),
    so we compare against tgt[:, 1:] as the ground truth.

    Parameters
    ----------
    logits  : (B, T-1, V)  — raw logits from the model forward pass
    tgt_ids : (B, T)       — full target sequence (includes leading SOS)
    pad_id  : int

    Returns
    -------
    dict with keys "token_acc", "sentence_acc", "correct_expression"
    """
    pred_ids = logits.argmax(dim=-1)   # (B, T-1)
    tgt_out  = tgt_ids[:, 1:]          # (B, T-1)  — drop leading SOS

    return {
        "token_acc":         token_accuracy(pred_ids, tgt_out, pad_id),
        "sentence_acc":      sentence_accuracy(pred_ids, tgt_out, pad_id),
        "correct_expression": correct_expression(pred_ids, pad_id),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PER-SEGMENT HELPERS  (unified multi-target model)
# ══════════════════════════════════════════════════════════════════════════════

def split_segments(
    ids:      List[int],
    break_id: int,
    eos_id:   int,
    pad_id:   int,
) -> List[List[int]]:
    """Split a flat token-ID list into per-coefficient segments.

    Traversal stops at the first EOS or PAD token.  Neither EOS, PAD, nor BREAK
    tokens appear in the returned segments.

    Parameters
    ----------
    ids      : flat token-ID list (e.g. the output of generate_batch, without SOS)
    break_id : ID of the <BREAK> delimiter token
    eos_id   : ID of <EOS>
    pad_id   : ID of <PAD>

    Returns
    -------
    List of segments, each a List[int] of raw token IDs.
    """
    segments: List[List[int]] = []
    current:  List[int]       = []
    for tid in ids:
        if tid == eos_id or tid == pad_id:
            break
        elif tid == break_id:
            segments.append(current)
            current = []
        else:
            current.append(tid)
    if current:                   # last segment before EOS/end-of-list
        segments.append(current)
    return segments


def _segment_correct_expression(pred_segments: List[List[int]]) -> float:
    """Correct-expression rate for pre-split segments (no EOS/PAD/BREAK)."""
    _FILTER = _SPECIAL | {"<BREAK>"}
    n_valid = 0
    for seg in pred_segments:
        toks = [
            VOCAB[tid]
            for tid in seg
            if 0 <= tid < len(VOCAB) and VOCAB[tid] not in _FILTER
        ]
        if _is_valid_prefix(toks):
            n_valid += 1
    return n_valid / len(pred_segments) if pred_segments else 0.0


def per_segment_metrics(
    all_pred:  List[List[int]],
    all_tgt:   List[List[int]],
    pad_id:    int,
    break_id:  int,
    n_coeffs:  int = N_COEFFS,
) -> List[Dict[str, float]]:
    """Compute token_acc, sentence_acc, and correct_expression for each segment.

    Parameters
    ----------
    all_pred  : N predicted sequences (each may contain BREAK tokens)
    all_tgt   : N ground-truth sequences (each may contain BREAK tokens)
    pad_id    : PAD token ID
    break_id  : BREAK token ID
    n_coeffs  : expected number of coefficient segments

    Returns
    -------
    List of ``n_coeffs`` dicts, each with keys
    "token_acc", "sentence_acc", "correct_expression".
    """
    eos_id = EOS_ID

    # Split every sequence into segments
    pred_segs = [split_segments(ids, break_id, eos_id, pad_id) for ids in all_pred]
    tgt_segs  = [split_segments(ids, break_id, eos_id, pad_id) for ids in all_tgt]

    results: List[Dict[str, float]] = []

    for coeff_i in range(n_coeffs):
        # Extract segment i (empty list if model generated fewer segments)
        preds_i = [
            segs[coeff_i] if coeff_i < len(segs) else []
            for segs in pred_segs
        ]
        tgts_i = [
            segs[coeff_i] if coeff_i < len(segs) else []
            for segs in tgt_segs
        ]

        # Pad to a common length for tensor metrics
        max_p = max((len(p) for p in preds_i), default=1)
        max_t = max((len(t) for t in tgts_i),  default=1)
        L = max(max_p, max_t)
        N = len(preds_i)

        pred_t = torch.full((N, L), pad_id, dtype=torch.long)
        tgt_t  = torch.full((N, L), pad_id, dtype=torch.long)
        for i, (p, t) in enumerate(zip(preds_i, tgts_i)):
            if p:
                pred_t[i, : len(p)] = torch.tensor(p, dtype=torch.long)
            if t:
                tgt_t[i,  : len(t)] = torch.tensor(t, dtype=torch.long)

        results.append({
            "token_acc":          token_accuracy(pred_t, tgt_t, pad_id),
            "sentence_acc":       sentence_accuracy(pred_t, tgt_t, pad_id),
            "correct_expression": _segment_correct_expression(preds_i),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PAD = 0

    # --- perfect prediction ---
    pred = torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tgt  = torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tok  = token_accuracy(pred, tgt, PAD)
    sent = sentence_accuracy(pred, tgt, PAD)
    assert tok  == 1.0, f"Expected 1.0, got {tok}"
    assert sent == 1.0, f"Expected 1.0, got {sent}"
    print(f"Perfect  : token_acc={tok:.4f}  sentence_acc={sent:.4f}")

    # --- one wrong token, different sequences ---
    pred = torch.tensor([[1, 9, 3, PAD], [4, 5, PAD, PAD]])
    tgt  = torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tok  = token_accuracy(pred, tgt, PAD)
    sent = sentence_accuracy(pred, tgt, PAD)
    # real tokens: seq0=[1,9,3] (3), seq1=[4,5] (2) → 5 total, 4 correct → 4/5
    # seq 0 wrong; seq 1 correct → sentence_acc = 0.5
    assert abs(tok  - 4/5) < 1e-6, f"Expected 0.8, got {tok}"
    assert abs(sent - 0.5) < 1e-6, f"Expected 0.5, got {sent}"
    print(f"One wrong: token_acc={tok:.4f}  sentence_acc={sent:.4f}")

    # --- all wrong ---
    pred = torch.tensor([[9, 9, 9, PAD], [9, 9, PAD, PAD]])
    tgt  = torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tok  = token_accuracy(pred, tgt, PAD)
    sent = sentence_accuracy(pred, tgt, PAD)
    assert tok  == 0.0
    assert sent == 0.0
    print(f"All wrong: token_acc={tok:.4f}  sentence_acc={sent:.4f}")

    # --- compute_metrics from logits ---
    B, T, V = 2, 5, 10
    # logits: (B, T-1, V) — predict tgt[:,1:]
    logits = torch.zeros(B, T - 1, V)
    tgt    = torch.tensor([[1, 3, 5, 7, PAD], [1, 2, 4, PAD, PAD]])
    # set logits so argmax matches tgt[:,1:]
    for b in range(B):
        for t in range(T - 1):
            logits[b, t, tgt[b, t + 1]] = 10.0
    m = compute_metrics(logits, tgt, PAD)
    assert m["token_acc"]    == 1.0
    assert m["sentence_acc"] == 1.0
    print(f"Logits   : token_acc={m['token_acc']:.4f}  sentence_acc={m['sentence_acc']:.4f}")

    print("\nAll smoke tests passed.")
