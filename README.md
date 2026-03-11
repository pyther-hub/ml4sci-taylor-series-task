# Taylor Series Coefficient Prediction

A deep learning system for predicting Taylor series coefficients of mathematical functions using Seq2Seq models (Transformer and LSTM).

## Setup & Installation

### Prerequisites
- Python 3.12+
- PyTorch (tested with CUDA support)
- SymPy for symbolic mathematics
- NumPy, Pandas, Scikit-learn, Matplotlib

### Installation Steps

1. **Clone/Navigate to the project directory:**
   ```bash
   cd /path/to/taylor-series-pred
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch sympy numpy pandas scikit-learn matplotlib jupyter
   ```

---

## Project Overview

This project trains neural networks to predict the **Taylor series coefficients** of mathematical functions. Given a function `f(x)` as input, the model outputs all 5 Taylor coefficients `(c0, c1, c2, c3, c4)` representing:

```
f(x) ~ c0 + c1*x + c2*x^2/2! + c3*x^3/3! + c4*x^4/4!
```

### Key Design
- **Single-pass decoding**: All coefficients predicted in one sequence
- **`<BREAK>` token delimiter**: Coefficients separated by special `<BREAK>` tokens in output sequence
- **Autoregressive generation**: Greedy decoding at inference

---

## Experiment Configurations

### 10k Dataset Run

**Environment:** Kaggle GPU (T4 16GB), 9-hour session limit

#### Shared Training Config

| Parameter | Value |
|-----------|-------|
| Dataset | `taylor_dataset_10k.json` (10,000 samples) |
| Train/Val Split | 90% / 10% |
| Random Seed | 1326 |
| Batch Size | 64 |
| Learning Rate | 3e-4 |
| Optimizer | Adam (default betas) |
| Scheduler | CosineAnnealingLR (eta_min = LR * 1e-2) |
| Max Epochs | 100 |
| Early Stopping | Patience = 10 epochs |
| Gradient Clipping | 1.0 |
| Loss Function | CrossEntropyLoss (ignore_index=PAD) |
| Max Seq Length | 512 |
| Vocab Size | 42 |

#### Transformer Config (`MODEL_TYPE = "transformer"`)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 8 |
| Encoder layers | 6 |
| Decoder layers | 8 |
| dim_feedforward | 256 |
| Dropout | 0.1 |
| Estimated params | ~4.3M |

#### LSTM Config (`MODEL_TYPE = "lstm"`)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| hidden_size | 256 |
| Encoder layers | 2 (bidirectional) |
| Decoder layers | 2 (unidirectional) |
| Attention | Bahdanau (additive) |
| Dropout | 0.1 |
| Weight tying | Yes (output projection = embedding) |

#### Demo Mode

Set `DEMO_RUN = True` to run only 2 epochs for quick testing/debugging.

---

## Model Architectures

### 1. Transformer Encoder-Decoder

```
Input Function (prefix tokens)
        |
   [SOS] + encode(fn_tokens) + [EOS]
        |
+----------------------+
|   ENCODER            |
| (6 layers)           |
| d_model=256          |
| nhead=8              |
| dim_feedforward=256  |
+----------------------+
        |
+----------------------+
|   DECODER            |
| (8 layers)           |
| Autoregressive       |
+----------------------+
        |
Output: [SOS] c0 [BREAK] c1 [BREAK] ... c4 [EOS]
```

- Sinusoidal positional encoding (max 512)
- Pre-norm transformer layers
- KV-cached greedy decoding

### 2. LSTM Seq2Seq with Attention

```
Input Function (prefix tokens)
        |
   [SOS] + encode(fn_tokens) + [EOS]
        |
+--------------------------+
|   ENCODER                |
| Bidirectional LSTM       |
| (2 layers, d=256, h=256)|
+--------------------------+
        |
   Bahdanau Attention
        |
+--------------------------+
|   DECODER                |
| Unidirectional LSTM      |
| (2 layers)               |
+--------------------------+
        |
Output: [SOS] c0 [BREAK] c1 [BREAK] ... c4 [EOS]
```

- Packed sequences for variable-length inputs
- Encoder final states projected to decoder initial states
- Context vector concatenated with decoder input at each step

**Shared across both models:**
- Same vocabulary (42 tokens) and special token handling
- Same public interface: `forward()`, `generate_batch()`, `generate()`
- Same checkpoint format

---

## Tokenization

### Vocabulary (42 tokens)

| Category | Tokens | Count |
|----------|--------|-------|
| Special | `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, `<BREAK>` | 5 |
| Variables | `x`, `a` | 2 |
| Operators | `+`, `-`, `*`, `/`, `**` | 5 |
| Digits | `0-9` | 10 |
| Functions | `sin`, `cos`, `exp`, `log`, `sqrt` | 5 |
| Delimiters | `(`, `)` | 2 |

Functions are encoded in **prefix notation** (Polish notation):
- `x^2 + 1` -> `[+, **, x, 2, 1]`

---

## Training Pipeline

### Training Loop

```
For each epoch (up to 100, with patience=10):
  1. Train on training set (teacher forcing)
  2. Validate on validation set (teacher forcing)
  3. Save checkpoint
  4. Track best val_loss, early stop if no improvement for 10 epochs

After training:
  5. Load best checkpoint
  6. Sample predictions on validation set
  7. Full autoregressive evaluation on validation set
  8. Generate all tables and figures
  9. Evaluate on 30 custom test functions (exact + SymPy match)
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Token Accuracy | Fraction of non-PAD tokens predicted correctly |
| Sentence Accuracy | Fraction of sequences where ALL tokens match |
| Expression Validity | Fraction of predictions that form valid prefix expressions |
| Function-Level Accuracy | Fraction where ALL 5 coefficients are correct |
| Per-Coefficient Accuracy | Token/sentence/expression accuracy for each c0-c4 |

### Custom Test Functions

30 handpicked functions evaluated at the end with both exact-match and SymPy equivalence checking. Examples:
- `(x**2 + 1)*sin(x)`, `exp(x)*cos(x)`, `log(1 + x)`, `x/(1 - x)`, etc.

---

## Output

### Generated Tables (CSV)

| File | Contents |
|------|----------|
| `architecture_summary.csv` | Model type, param count, layer config |
| `performance_summary.csv` | Final losses, accuracies, training info |
| `per_coefficient_accuracy.csv` | c0-c4 token/sentence/expression accuracy |
| `full_run_metrics.csv` | Per-epoch: loss, token acc, sentence acc |
| `eval_functions_results.csv` | Per-function exact/sympy match results |

### Generated Figures (PNG)

| File | Contents |
|------|----------|
| `fig1_loss_curves.png` | Train/val loss over epochs |
| `fig2_sentence_accuracy.png` | Train/val sentence accuracy over epochs |
| `fig3_per_coefficient_accuracy.png` | Grouped bar chart: c0-c4 accuracy |
| `fig4_token_accuracy.png` | Train/val token accuracy over epochs |
| `fig5_sequence_lengths.png` | Predicted vs ground truth length histogram |

All outputs are saved to `reports/{model_type}_{timestamp}/`.

---

## Project Structure

```
taylor-series-pred/
├── main.py                  # Training entry point + report generation
├── model.py                 # CoeffPredTransformer & CoeffPredLSTM
├── dataset.py               # Dataset class & vocabulary
├── dataset_generation.py    # Dataset generation utilities
├── train_validate.py        # Training/validation loops
├── metrics.py               # Evaluation metrics
├── report_logger.py         # Report logger with plot generation
├── inference.py             # Inference utilities
├── checkpoints/             # Saved model checkpoints
│   ├── epoch_NNN.pt
│   └── taylor_series_pred_{model_type}.pt
├── reports/                 # Generated reports (tables + figures)
│   └── {model_type}_{timestamp}/
└── *.ipynb                  # Jupyter notebooks
```

---

## Quick Start

```bash
# Full training run
python main.py  # Edit MODEL_TYPE in main.py: "transformer" or "lstm"

# Demo run (2 epochs)
# Set DEMO_RUN = True in main.py, then:
python main.py
```

### Customization

Edit the configuration section in `main.py`:

```python
DEMO_RUN = False          # True for 2-epoch test run
MODEL_TYPE = "transformer"  # "transformer" or "lstm"
RANDOM_SEED = 1326
NUM_EPOCHS = 100
PATIENCE = 10             # Early stopping patience
BATCH_SIZE = 64
LR = 3e-4
```

---

## Notes

- Prefix notation (Polish notation) for all expressions
- `<BREAK>` tokens delimit coefficients in the output sequence
- Teacher forcing during training; autoregressive decoding for evaluation
- Sequences > 512 tokens are filtered out
- Best checkpoint selected by validation loss
- Early stopping with patience=10 prevents overfitting
