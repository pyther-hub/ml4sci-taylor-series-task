"""
train_validate.py
=================
Training and validation utilities for the Taylor coefficient prediction pipeline.

Functions
---------
  set_seed(seed)
  get_device()
  build_dataloaders(json_path, val_ratio, seed, batch_size, num_workers)
  train_epoch(model, loader, optimizer, criterion, device, clip_grad)
  validate(model, loader, criterion, device)
  print_epoch(epoch, num_epochs, train_m, val_m, elapsed, is_best)
"""

from __future__ import annotations

import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import VOCAB_SIZE, PAD_ID, CoeffPredDataset
from metrics import compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    json_path:   str,
    val_ratio:   float,
    seed:        int,
    batch_size:  int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, CoeffPredDataset]:
    """Load the dataset and return train / val DataLoaders + the full dataset."""
    full_ds = CoeffPredDataset(json_path)
    n_val   = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=full_ds.collate_fn,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=full_ds.collate_fn,
    )
    return train_loader, val_loader, full_ds


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    clip_grad: float,
) -> Dict[str, float]:
    """One full training epoch with teacher-forced loss and greedy-decode metrics.

    Progress is printed every 10 % of steps showing steps left and time/step.
    """
    model.train()

    total_loss  = 0.0
    total_tok   = 0.0
    total_sent  = 0.0
    n_steps     = len(loader)
    print_every = max(1, n_steps // 10)

    for step, (src, tgt, _, _) in enumerate(loader, 1):
        t_step = time.perf_counter()

        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt)                                    # (B, T-1, V)
        loss   = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt[:, 1:].reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        m = compute_metrics(logits.detach(), tgt, PAD_ID)
        total_loss += loss.item()
        total_tok  += m["token_acc"]
        total_sent += m["sentence_acc"]

        if step % print_every == 0 or step == n_steps:
            step_time = time.perf_counter() - t_step
            left      = n_steps - step
            print(
                f"  [train] step {step:4d}/{n_steps}"
                f"  left={left:4d}"
                f"  loss={total_loss / step:.4f}"
                f"  tok={total_tok / step:.3f}"
                f"  sent={total_sent / step:.3f}"
                f"  ({step_time:.2f}s/step)"
            )

    return {
        "train_loss":      total_loss / n_steps,
        "train_tok_acc":   total_tok  / n_steps,
        "train_sent_acc":  total_sent / n_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, float]:
    """Teacher-forced validation: cross-entropy loss + greedy-decode accuracy."""
    model.eval()

    total_loss  = 0.0
    total_tok   = 0.0
    total_sent  = 0.0
    n_steps     = len(loader)

    with torch.no_grad():
        for src, tgt, _, _ in loader:
            src, tgt = src.to(device), tgt.to(device)

            logits = model(src, tgt)                                # (B, T-1, V)
            loss   = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt[:, 1:].reshape(-1),
            )
            m = compute_metrics(logits, tgt, PAD_ID)

            total_loss += loss.item()
            total_tok  += m["token_acc"]
            total_sent += m["sentence_acc"]

    return {
        "val_loss":      total_loss / n_steps,
        "val_tok_acc":   total_tok  / n_steps,
        "val_sent_acc":  total_sent / n_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_epoch(
    epoch:      int,
    num_epochs: int,
    train_m:    Dict[str, float],
    val_m:      Dict[str, float],
    elapsed:    float,
    is_best:    bool,
) -> None:
    star = "  *best*" if is_best else ""
    print(
        f"  epoch {epoch:3d}/{num_epochs}"
        f"  trn_loss={train_m['train_loss']:.4f}"
        f"  val_loss={val_m['val_loss']:.4f}"
        f"  val_tok={val_m['val_tok_acc']:.3f}"
        f"  val_sent={val_m['val_sent_acc']:.3f}"
        f"  {elapsed:.1f}s"
        f"{star}"
    )
