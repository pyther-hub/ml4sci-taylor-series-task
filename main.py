"""
main.py
==================
Training entry point for the unified Taylor coefficient prediction pipeline.

The model predicts ALL five Taylor coefficients in a single pass, using
<BREAK> tokens to delimit individual coefficients in the output sequence.

Training loop only computes teacher-forced val loss (fast).  After training,
evaluate_checkpoints() loads each saved epoch checkpoint, runs greedy decoding,
and computes full accuracy metrics.

Usage
-----
python main.py

Edit the hyperparameters in the ``if __name__ == "__main__":`` block at the
bottom of this file.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import torch
import torch.nn as nn

from dataset import BREAK_ID, EOS_ID, N_COEFFS, PAD_ID, SOS_ID, VOCAB_SIZE, decode, encode
from model import CoeffPredLSTM, CoeffPredTransformer
from train_validate import (
build_dataloaders,
get_device,
print_epoch,
set_seed,
train_epoch,
validate,
)

# ── Evaluation on fixed function list ─────────────────────────────────────
import sympy as sp
from dataset_generation import (
    _expr_to_prefix_tokens,
    compute_taylor_coefficients,
    prefix_tokens_to_infix,
)
from metrics import split_segments

EVAL_FUNCTIONS = [
    "(x**2 + 1)*sin(x)",
    "x**3*cos(2*x)",
    "(2*x + 1)*sin(3*x)",
    "(x**2 - x)*cos(x)",
    "exp(x)*(1 + x)",
    "x*exp(2*x)",
    "(1 + x**2)*exp(-x)",
    "exp(x)*sin(x)",
    "log(1 + x)",
    "log(1 + x**2)",
    "x*log(1 + x)",
    "log(1 + 2*x + x**2)",
    "x/(1 - x)",
    "x/(1 + x)",
    "1/(1 + x**2)",
    "x**2/(1 - x)",
    "sin(x**2)",
    "exp(x**2)",
    "cos(sqrt(1 + x))",
    "log(1 + sin(x))",
    "(x + 1)*exp(x)",
    "sin(x)/(1 + x)",
    "x**2*log(1 + x)",
    "exp(x)*cos(x)",
    "(x**2 + 2*x + 1)*sin(x)",
    "(x + 2)*cos(2*x)",
    "exp(x)*(x**2 + 1)",
    "(x**2 + 1)*exp(2*x)",
    "sin(x)*cos(x)",
    "sin(2*x)/(1 + x)",
    "cos(x)/(1 + x**2)",
    "(x + 1)/(1 + x**2)",
    "(x**2 + x)/(1 + x)",
    "(x**2 + 1)/(1 + x)",
    "log(1 + x + x**2)",
    "log(1 + x**3)",
    "log(1 + x)*sin(x)",
    "log(1 + x)*cos(x)",
    "x*exp(x)*sin(x)",
    "x*exp(x)*cos(x)",
    "exp(x)*log(1 + x)",
    "exp(x)*sqrt(1 + x)",
    "sqrt(1 + x)*sin(x)",
    "sqrt(1 + x)*cos(x)",
    "sqrt(1 + x)/(1 + x)",
    "sin(x)/(1 + x**2)",
    "cos(x)/(1 + x)",
    "(x + 1)*sin(x)*cos(x)",
    "(x**2 + 1)*sin(2*x)",
    "(x**2 + x + 1)*cos(x)",
    "(x + 1)*log(1 + x)",
    "(x**2 + 1)*log(1 + x)",
    "exp(x)/(1 + x)",
    "exp(x)/(1 + x**2)",
    "sin(x**2)/(1 + x)",
    "cos(x**2)/(1 + x)",
    "exp(x**2)/(1 + x)",
    "log(1 + x**2)*sin(x)",
    "log(1 + x**2)*cos(x)",
    "sin(x)*exp(x**2)",
    "cos(x)*exp(x**2)",
]

TAYLOR_ORDER = N_COEFFS - 1   # 4 → coefficients c0 … c4
x_sym        = sp.Symbol("x")



# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CLASSES = {
    "transformer": CoeffPredTransformer,
    "lstm":        CoeffPredLSTM,
}

def build_model(model_type: str, config: Dict) -> nn.Module:
    """Instantiate the correct model class from model_type and config dict."""
    cls = MODEL_CLASSES[model_type]
    return cls(**config)


def save_checkpoint(
    model:      nn.Module,
    path:       str,
    epoch:      int,
    val_metrics: Dict,
    arch_cfg:   Dict,
    model_type: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch":       epoch,
            "n_coeffs":    N_COEFFS,
            "val_loss":    val_metrics.get("val_loss", float("inf")),
            "model_state": model.state_dict(),
            "config":      arch_cfg,
            "model_type":  model_type,
        },
        path,
    )
    print(f"  [save] checkpoint → {path}  (epoch={epoch})")



# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_TYPE = "transformer"  # "transformer" or "lstm"

TRANSFORMER_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 8,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "max_seq_len": 512,
}

LSTM_CONFIG = {
    "d_model": 256,
    "hidden_size": 256,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dropout": 0.1,
    "max_seq_len": 512,
}

# ── Paths ─────────────────────────────────────────────────────────────────
DATASET_JSON    = os.path.join("", "/kaggle/input/datasets/tensorpanda231/taylor-series-dataset-simple/taylor_dataset_10k.json")
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"taylor_series_pred_{MODEL_TYPE}.pt")


# ── Data split ────────────────────────────────────────────────────────────
VAL_RATIO   = 0.10
RANDOM_SEED = 42

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_EPOCHS  = 100
LR          = 3e-4
CLIP_GRAD   = 1.0
NUM_WORKERS = 0

# ── Post-training evaluation ──────────────────────────────────────────────
MAX_GEN_LEN = 512          # max decode steps for post-training greedy eval
EVALUATE_ON_EVAL_FUNCTIONS_AFTER = 5  # run eval on EVAL_FUNCTIONS every N epochs



# ─────────────────────────────────────────────────────────────────────────
set_seed(RANDOM_SEED)
device = get_device()
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 68)
print(f"  coeff_pred training — unified model ({N_COEFFS} coefficients)")
print(f"  device  : {device}")
print(f"  dataset : {DATASET_JSON}")
print("=" * 68)

# ── Data ──────────────────────────────────────────────────────────────────
train_loader, val_loader, full_ds = build_dataloaders(
    DATASET_JSON, VAL_RATIO, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
    max_seq_len=MAX_SEQ_LEN,
)
print(f"\n  Dataset  : {len(full_ds)} items  (skipped={full_ds.n_skipped})")
print(f"  Train    : {len(train_loader.dataset)}   Val : {len(val_loader.dataset)}\n")

# ── Model ─────────────────────────────────────────────────────────────────
arch_cfg = TRANSFORMER_CONFIG if MODEL_TYPE == "transformer" else LSTM_CONFIG
model    = build_model(MODEL_TYPE, arch_cfg).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Params   : {n_params:,}   VOCAB_SIZE={VOCAB_SIZE}\n")

# ── Optimizer / loss / scheduler ──────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=LR * 1e-2,
)

# ── Training loop (no greedy decode — fast) ──────────────────────────────
best_val_loss = float("inf")


for epoch in range(1, NUM_EPOCHS + 1):
    t0      = time.perf_counter()
    train_m = train_epoch(model, train_loader, optimizer, criterion, device, CLIP_GRAD)
    val_m   = validate(model, val_loader, criterion, device)
    scheduler.step()

    elapsed = time.perf_counter() - t0
    is_best = val_m["val_loss"] < best_val_loss

    # Save checkpoint every epoch (for post-training evaluation).
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    val_m["val_loss"],
            "n_coeffs":    N_COEFFS,
            "config":      arch_cfg,
            "model_type":  MODEL_TYPE,
        },
        ckpt_path,
    )

    if is_best:
        best_val_loss = val_m["val_loss"]
        save_checkpoint(model, CHECKPOINT_PATH, epoch, val_m, arch_cfg, MODEL_TYPE)

    print_epoch(epoch, NUM_EPOCHS, train_m, val_m, elapsed, is_best)

    if epoch % EVALUATE_ON_EVAL_FUNCTIONS_AFTER == 0:

        print("\n" + "=" * 68)
        print("  Evaluation on fixed function list (autoregressive decoding)")
        print("=" * 68)
        
        n_fn_correct  = 0
        n_fn_attempted = 0
        
        for fn_idx, fn_str in enumerate(EVAL_FUNCTIONS, 1):
        
            # 1. Parse string → sympy
            try:
                expr = sp.sympify(fn_str, locals={"x": x_sym})
            except Exception as e:
                print(f"\n  [{fn_idx:2d}] {fn_str}")
                print(f"       PARSE ERROR: {e}")
                continue
        
            # 2. Ground-truth Taylor coefficients (nth derivative at x=a)
            gt_coeffs = compute_taylor_coefficients(expr, TAYLOR_ORDER)
            if gt_coeffs is None:
                print(f"\n  [{fn_idx:2d}] {fn_str}")
                print(f"       COEFF ERROR: sympy timed out")
                continue
        
            gt_token_lists = [_expr_to_prefix_tokens(c) for c in gt_coeffs]
        
            # 3. Encode source sequence
            fn_prefix = _expr_to_prefix_tokens(expr)
            src_ids   = [SOS_ID] + encode(fn_prefix) + [EOS_ID]
            src       = torch.tensor([src_ids], dtype=torch.long).to(device)
        
            # 4. Autoregressive decoding — one token at a time, predicted token
            #    fed back as next input (implemented inside model.generate)
            pred_ids = model.generate(src, max_len=MAX_GEN_LEN)   # List[int]
        
            # 5. Split flat prediction on <BREAK> into per-coefficient segments
            pred_segs = split_segments(pred_ids, BREAK_ID, EOS_ID, PAD_ID)
        
            # 6. Compare and display
            n_fn_attempted += 1
            n_coeff_correct = 0
        
            print(f"\n  [{fn_idx:2d}] f(x) = {fn_str}")
            for coeff_i in range(N_COEFFS):
                gt_toks   = gt_token_lists[coeff_i] if coeff_i < len(gt_token_lists) else []
                pred_toks = pred_segs[coeff_i]       if coeff_i < len(pred_segs)     else []
        
                gt_ids    = encode(gt_toks)           # List[int]
                match     = (gt_ids == pred_toks)
        
                gt_infix   = prefix_tokens_to_infix(gt_toks)
                pred_infix = prefix_tokens_to_infix(decode(pred_toks, skip_special=False))
        
                status = "OK" if match else "--"
                print(
                    f"       c{coeff_i} [{status}]"
                    f"  gt={gt_infix}"
                    f"  pred={pred_infix}"
                )
                if match:
                    n_coeff_correct += 1
        
            all_correct = (n_coeff_correct == N_COEFFS)
            if all_correct:
                n_fn_correct += 1
            print(f"       => {n_coeff_correct}/{N_COEFFS} coefficients correct")
        
        print("\n" + "=" * 68)
        print(
            f"  Result : {n_fn_correct}/{n_fn_attempted} functions"
            f" with all {N_COEFFS} coefficients correct"
            f"  ({100*n_fn_correct/n_fn_attempted:.1f}%)" if n_fn_attempted else ""
        )
        print("=" * 68)

        

print(f"\n  Training complete.  Best val loss: {best_val_loss:.6f}")
print(f"  Best checkpoint: {CHECKPOINT_PATH}")

# ── Sample predictions from best checkpoint ──────────────────────────────
print("\n  Loading best model for example predictions ...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"\n  Example predictions (unified, greedy):")
shown = 0
with torch.no_grad():
    for src, tgt, _, _ in val_loader:
        src   = src.to(device)
        preds = model.generate_batch(src, max_len=MAX_GEN_LEN)
        for i in range(len(preds)):
            if shown >= 4:
                break
            src_tok  = decode(src[i].tolist())
            tgt_tok  = decode(tgt[i].tolist(), skip_special=False)
            pred_tok = decode([t for t in preds[i] if t != EOS_ID], skip_special=False)
            print(f"  ── example {shown + 1} ──")
            print(f"     src  : {src_tok}")
            print(f"     tgt  : {tgt_tok}")
            print(f"     pred : {pred_tok}")
            print(f"     match: {'✓' if tgt_tok == pred_tok else '✗'}")
            shown += 1
        if shown >= 4:
            break


# ── Evaluation on fixed function list ─────────────────────────────────────
import sympy as sp
from dataset_generation import (
    _expr_to_prefix_tokens,
    compute_taylor_coefficients,
    prefix_tokens_to_infix,
)
from metrics import split_segments


TAYLOR_ORDER = N_COEFFS - 1   # 4 → coefficients c0 … c4
x_sym        = sp.Symbol("x")

print("\n" + "=" * 68)
print("  Evaluation on fixed function list (autoregressive decoding)")
print("=" * 68)

n_fn_correct  = 0
n_fn_attempted = 0

for fn_idx, fn_str in enumerate(EVAL_FUNCTIONS, 1):

    # 1. Parse string → sympy
    try:
        expr = sp.sympify(fn_str, locals={"x": x_sym})
    except Exception as e:
        print(f"\n  [{fn_idx:2d}] {fn_str}")
        print(f"       PARSE ERROR: {e}")
        continue

    # 2. Ground-truth Taylor coefficients (nth derivative at x=a)
    gt_coeffs = compute_taylor_coefficients(expr, TAYLOR_ORDER)
    if gt_coeffs is None:
        print(f"\n  [{fn_idx:2d}] {fn_str}")
        print(f"       COEFF ERROR: sympy timed out")
        continue

    gt_token_lists = [_expr_to_prefix_tokens(c) for c in gt_coeffs]

    # 3. Encode source sequence
    fn_prefix = _expr_to_prefix_tokens(expr)
    src_ids   = [SOS_ID] + encode(fn_prefix) + [EOS_ID]
    src       = torch.tensor([src_ids], dtype=torch.long).to(device)

    # 4. Autoregressive decoding — one token at a time, predicted token
    #    fed back as next input (implemented inside model.generate)
    pred_ids = model.generate(src, max_len=MAX_GEN_LEN)   # List[int]

    # 5. Split flat prediction on <BREAK> into per-coefficient segments
    pred_segs = split_segments(pred_ids, BREAK_ID, EOS_ID, PAD_ID)

    # 6. Compare and display
    n_fn_attempted += 1
    n_coeff_correct = 0

    print(f"\n  [{fn_idx:2d}] f(x) = {fn_str}")
    for coeff_i in range(N_COEFFS):
        gt_toks   = gt_token_lists[coeff_i] if coeff_i < len(gt_token_lists) else []
        pred_toks = pred_segs[coeff_i]       if coeff_i < len(pred_segs)     else []

        gt_ids    = encode(gt_toks)           # List[int]
        match     = (gt_ids == pred_toks)

        gt_infix   = prefix_tokens_to_infix(gt_toks)
        pred_infix = prefix_tokens_to_infix(decode(pred_toks, skip_special=False))

        status = "OK" if match else "--"
        print(
            f"       c{coeff_i} [{status}]"
            f"  gt={gt_infix}"
            f"  pred={pred_infix}"
        )
        if match:
            n_coeff_correct += 1

    all_correct = (n_coeff_correct == N_COEFFS)
    if all_correct:
        n_fn_correct += 1
    print(f"       => {n_coeff_correct}/{N_COEFFS} coefficients correct")

print("\n" + "=" * 68)
print(
    f"  Result : {n_fn_correct}/{n_fn_attempted} functions"
    f" with all {N_COEFFS} coefficients correct"
    f"  ({100*n_fn_correct/n_fn_attempted:.1f}%)" if n_fn_attempted else ""
)
print("=" * 68)
