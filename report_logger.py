"""
report_logger.py
================
Class-based report logger for capturing training metrics, evaluation results,
and generating plots / structured output files.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import sympy as sp

from dataset_generation import _prefix_tokens_to_expr


# ══════════════════════════════════════════════════════════════════════════════
# SymPy equivalence helper
# ══════════════════════════════════════════════════════════════════════════════

def sympy_equiv(pred_tokens: List[str], gt_expr: sp.Expr, timeout_sec: float = 3.0) -> bool:
    """Check if predicted prefix tokens are mathematically equivalent to gt_expr.

    Parameters
    ----------
    pred_tokens : prefix token strings (no special tokens like SOS/EOS/PAD)
    gt_expr     : ground-truth SymPy expression
    timeout_sec : not enforced here (sympy simplify is usually fast for scalars)

    Returns
    -------
    True if simplify(pred - gt) == 0, False on any error.
    """
    clean = [t for t in pred_tokens if t not in ("<SOS>", "<EOS>", "<PAD>", "<BREAK>")]
    if not clean:
        return False
    try:
        pred_expr, _ = _prefix_tokens_to_expr(clean, 0)
        diff = sp.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ReportLogger
# ══════════════════════════════════════════════════════════════════════════════

class ReportLogger:
    """Accumulates training metrics and generates a structured report."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.config: Dict[str, Any] = {}
        self.epoch_logs: List[Dict[str, Any]] = []
        self.val_eval_results: Optional[Dict[str, float]] = None
        self.per_coeff_results: Optional[List[Dict[str, float]]] = None
        self.eval_function_results: List[Dict[str, Any]] = []
        self.seq_lengths: Optional[Dict[str, List[int]]] = None  # pred_lengths, gt_lengths

    # ── Logging methods ──────────────────────────────────────────────────

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration (called once before training)."""
        self.config = config
        path = os.path.join(self.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log per-epoch metrics."""
        entry = {"epoch": epoch, **metrics}
        self.epoch_logs.append(entry)

    def log_val_eval(self, results: Dict[str, float]) -> None:
        """Log final validation set evaluation results."""
        self.val_eval_results = results

    def log_per_coefficient_accuracy(self, per_coeff: List[Dict[str, float]]) -> None:
        """Log per-coefficient (c0-c4) accuracy results."""
        self.per_coeff_results = per_coeff

    def log_eval_function_result(
        self,
        fn_str: str,
        coeff_results: List[Dict[str, Any]],
        all_exact_match: bool,
        all_sympy_match: bool,
    ) -> None:
        """Log result for a single EVAL_FUNCTION."""
        self.eval_function_results.append({
            "function": fn_str,
            "coeff_results": coeff_results,
            "all_exact_match": all_exact_match,
            "all_sympy_match": all_sympy_match,
        })

    def log_sequence_lengths(self, pred_lengths: List[int], gt_lengths: List[int]) -> None:
        """Log predicted vs ground-truth sequence lengths for histogram."""
        self.seq_lengths = {"pred": pred_lengths, "gt": gt_lengths}

    # ── Report generation ────────────────────────────────────────────────

    def generate_report(self) -> None:
        """Write all outputs: CSVs, JSON summary, and plots."""
        print(f"\n  Generating report in {self.output_dir}/ ...")

        if self.epoch_logs:
            self._save_epoch_csv()
            self._plot_loss_curves()
            self._plot_accuracy_curves()

        if self.eval_function_results:
            self._save_eval_functions_table()

        if self.per_coeff_results:
            self._plot_per_coefficient_bar_chart()

        if self.seq_lengths:
            self._plot_sequence_length_histogram()

        self._save_summary_json()
        print(f"  Report complete: {self.output_dir}/")

    # ── Private: data export ─────────────────────────────────────────────

    def _save_epoch_csv(self) -> None:
        path = os.path.join(self.output_dir, "epoch_logs.csv")
        fields = ["epoch", "train_loss", "val_loss",
                  "train_tok_acc", "val_tok_acc",
                  "train_sent_acc", "val_sent_acc"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.epoch_logs)

    def _save_eval_functions_table(self) -> None:
        path = os.path.join(self.output_dir, "eval_functions_results.csv")
        fields = ["function"]
        for i in range(5):
            fields += [f"c{i}_gt", f"c{i}_pred", f"c{i}_exact", f"c{i}_sympy"]
        fields += ["all_exact", "all_sympy"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for entry in self.eval_function_results:
                row: Dict[str, Any] = {"function": entry["function"]}
                for cr in entry["coeff_results"]:
                    i = cr["coeff_idx"]
                    row[f"c{i}_gt"] = cr.get("gt_infix", "")
                    row[f"c{i}_pred"] = cr.get("pred_infix", "")
                    row[f"c{i}_exact"] = cr.get("exact_match", False)
                    row[f"c{i}_sympy"] = cr.get("sympy_match", False)
                row["all_exact"] = entry["all_exact_match"]
                row["all_sympy"] = entry["all_sympy_match"]
                writer.writerow(row)

    def _save_summary_json(self) -> None:
        summary: Dict[str, Any] = {"config": self.config}

        if self.val_eval_results:
            summary["val_eval"] = self.val_eval_results

        if self.per_coeff_results:
            summary["per_coefficient"] = [
                {"coeff": f"c{i}", **d}
                for i, d in enumerate(self.per_coeff_results)
            ]

        if self.eval_function_results:
            n = len(self.eval_function_results)
            n_exact = sum(1 for e in self.eval_function_results if e["all_exact_match"])
            n_sympy = sum(1 for e in self.eval_function_results if e["all_sympy_match"])
            summary["eval_functions"] = {
                "n_functions": n,
                "exact_match_all_correct": n_exact,
                "exact_match_accuracy": round(n_exact / n, 4) if n else 0,
                "sympy_match_all_correct": n_sympy,
                "sympy_match_accuracy": round(n_sympy / n, 4) if n else 0,
            }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # ── Private: plots ───────────────────────────────────────────────────

    def _plot_loss_curves(self) -> None:
        epochs = [e["epoch"] for e in self.epoch_logs]
        train_loss = [e["train_loss"] for e in self.epoch_logs]
        val_loss = [e["val_loss"] for e in self.epoch_logs]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
        ax.plot(epochs, val_loss, label="Val Loss", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.output_dir, "loss_curves.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_accuracy_curves(self) -> None:
        epochs = [e["epoch"] for e in self.epoch_logs]
        train_acc = [e["train_tok_acc"] for e in self.epoch_logs]
        val_acc = [e["val_tok_acc"] for e in self.epoch_logs]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_acc, label="Train Token Acc", color="tab:blue")
        ax.plot(epochs, val_acc, label="Val Token Acc", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Token Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.output_dir, "accuracy_curves.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_per_coefficient_bar_chart(self) -> None:
        if not self.per_coeff_results:
            return

        import numpy as np

        n_coeffs = len(self.per_coeff_results)
        labels = [f"c{i}" for i in range(n_coeffs)]
        tok_acc = [d["token_acc"] for d in self.per_coeff_results]
        sent_acc = [d["sentence_acc"] for d in self.per_coeff_results]
        expr_valid = [d["correct_expression"] for d in self.per_coeff_results]

        x = np.arange(n_coeffs)
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, tok_acc, width, label="Token Acc", color="tab:blue")
        ax.bar(x, sent_acc, width, label="Sentence Acc", color="tab:green")
        ax.bar(x + width, expr_valid, width, label="Valid Expression", color="tab:red")

        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Coefficient Accuracy (c0 - c4)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(os.path.join(self.output_dir, "per_coefficient_accuracy.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_sequence_length_histogram(self) -> None:
        if not self.seq_lengths:
            return

        pred = self.seq_lengths["pred"]
        gt = self.seq_lengths["gt"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(gt, bins=30, alpha=0.6, label="Ground Truth", color="tab:blue")
        ax.hist(pred, bins=30, alpha=0.6, label="Predicted", color="tab:orange")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Count")
        ax.set_title("Predicted vs Ground Truth Sequence Lengths")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.output_dir, "sequence_length_histogram.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
