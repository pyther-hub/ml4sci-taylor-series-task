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

from dataset import EOS_ID, N_COEFFS, PAD_ID, VOCAB_SIZE, decode
from model import CoeffPredTransformer
from train_validate import (
    build_dataloaders,
    evaluate_checkpoints,
    get_device,
    print_epoch,
    set_seed,
    train_epoch,
    validate,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    d_model:            int,
    nhead:              int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_feedforward:    int,
    dropout:            float,
    max_seq_len:        int,
) -> CoeffPredTransformer:
    return CoeffPredTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )


def save_checkpoint(
    model:       CoeffPredTransformer,
    path:        str,
    epoch:       int,
    val_metrics: Dict,
    arch_cfg:    Dict,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch":       epoch,
            "n_coeffs":    N_COEFFS,
            "val_loss":    val_metrics.get("val_loss", float("inf")),
            "model_state": model.state_dict(),
            "config":      arch_cfg,
        },
        path,
    )
    print(f"  [save] checkpoint → {path}  (epoch={epoch})")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATASET_JSON    = os.path.join("datasets", "taylor_dataset_cleaned_4.json")
    CHECKPOINT_DIR  = "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unified_best.pt")

    # ── Data split ────────────────────────────────────────────────────────────
    VAL_RATIO   = 0.10
    RANDOM_SEED = 42

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE  = 32
    NUM_EPOCHS  = 50
    LR          = 3e-4
    CLIP_GRAD   = 1.0
    NUM_WORKERS = 0

    # ── Model architecture ────────────────────────────────────────────────────
    D_MODEL            = 128
    NHEAD              = 4      # D_MODEL must be divisible by NHEAD
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD    = 256
    DROPOUT            = 0.1
    MAX_SEQ_LEN        = 1024

    # ── Post-training evaluation ──────────────────────────────────────────────
    MAX_GEN_LEN = 768          # max decode steps for post-training greedy eval

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
    )
    print(f"\n  Dataset  : {len(full_ds)} items  (skipped={full_ds.n_skipped})")
    print(f"  Train    : {len(train_loader.dataset)}   Val : {len(val_loader.dataset)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    arch_cfg = dict(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    )
    model    = build_model(**arch_cfg).to(device)
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
            },
            ckpt_path,
        )

        if is_best:
            best_val_loss = val_m["val_loss"]
            save_checkpoint(model, CHECKPOINT_PATH, epoch, val_m, arch_cfg)

        print_epoch(epoch, NUM_EPOCHS, train_m, val_m, elapsed, is_best)

    print(f"\n  Training complete.  Best val loss: {best_val_loss:.6f}")
    print(f"  Best checkpoint: {CHECKPOINT_PATH}")

    # ── Post-training evaluation (greedy decode all epoch checkpoints) ────────
    results = evaluate_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        model=model,
        val_loader=val_loader,
        device=device,
        max_gen=MAX_GEN_LEN,
        save_path=os.path.join(CHECKPOINT_DIR, "all_epoch_predictions.npz"),
    )

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
