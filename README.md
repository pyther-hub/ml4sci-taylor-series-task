# Taylor Series Coefficient Prediction

A deep learning system for predicting Taylor series coefficients of mathematical functions using a Sequence-to-Sequence Transformer model.

## 📋 Setup & Installation

### Prerequisites
- Python 3.12+
- PyTorch (tested with CUDA support)
- SymPy for symbolic mathematics
- NumPy, Pandas, Scikit-learn

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
   pip install torch sympy numpy pandas scikit-learn jupyter
   ```

4. **Verify setup:**
   ```bash
   python main.py --help  # Or just python main.py to start training
   ```

---

## 🎯 Project Overview

This project trains a Transformer-based neural network to predict the **Taylor series coefficients** of mathematical functions. Given a function `f(x)` as input, the model outputs all 5 Taylor coefficients `(c₀, c₁, c₂, c₃, c₄)` representing:

```
f(x) ≈ c₀ + c₁·x/1 + c₂·x²/2 + c₃·x³/6 + c₄·x⁴/24
```

### Key Innovation: Unified Prediction
- **Single-pass decoding**: All coefficients predicted in one sequence instead of 5 separate models
- **<BREAK> token delimiter**: Coefficients separated by special `<BREAK>` tokens in output sequence
- **Autoregressive generation**: One token at a time with greedy decoding

---

## 🔤 Tokenization Strategy

### Vocabulary (42 tokens)

The model uses a fixed vocabulary comprising:

| Category | Tokens | Count |
|----------|--------|-------|
| **Special** | `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, `<BREAK>` | 5 |
| **Variables** | `x`, `a` | 2 |
| **Operators** | `+`, `-`, `*`, `/`, `**` | 5 |
| **Digits** | `0-9` | 10 |
| **Functions** | `sin`, `cos`, `exp`, `log`, `sqrt` | 5 |
| **Delimiters** | `(`, `)` | 2 |
| **Math Constants** | (represented as digits/fractions) | - |

**Total Vocabulary Size: 42 tokens**

### Encoding Scheme

Functions are represented in **prefix notation** (Polish notation):
- Example: `x² + 1` → `[+, **, x, 2, 1]`
- Unknown tokens default to `<UNK>` token

### Special Tokens

| Token | ID | Purpose |
|-------|----|----|
| `<PAD>` | 0 | Padding for variable-length sequences |
| `<SOS>` | 1 | Start of Sequence (input/output) |
| `<EOS>` | 2 | End of Sequence |
| `<UNK>` | 3 | Unknown tokens |
| `<BREAK>` | 27 | Delimiter between coefficient segments |

---

## 📊 Model Architectures

The project supports two model architectures, selectable via `MODEL_TYPE` in `main.py`.

### 1. Transformer Encoder-Decoder (`MODEL_TYPE = "transformer"`)

```
Input Function (prefix tokens)
        ↓
   [SOS] + encode(fn_tokens) + [EOS]
        ↓
┌──────────────────────┐
│   ENCODER            │
│ (6 layers)           │
│ d_model=256          │
│ nhead=8              │
│ dim_feedforward=256  │
└──────────────────────┘
        ↓
┌──────────────────────┐
│   DECODER            │
│ (8 layers)           │
│ Autoregressive       │
└──────────────────────┘
        ↓
Output Tokens (coefficients separated by <BREAK>)
[SOS] + c0_tokens + [BREAK] + c1_tokens + [BREAK] + ... + [EOS]
```

**Architecture Details:**
- Embedding dimension: 256, Attention heads: 8
- Encoder: 6 layers, Decoder: 8 layers
- Feedforward hidden: 256, Dropout: 0.1
- Positional encoding: Sinusoidal (max seq len: 512)
- Total parameters: ~4.3M

### 2. LSTM Seq2Seq with Attention (`MODEL_TYPE = "lstm"`)

```
Input Function (prefix tokens)
        ↓
   [SOS] + encode(fn_tokens) + [EOS]
        ↓
┌──────────────────────────┐
│   ENCODER                │
│ Bidirectional LSTM       │
│ (2 layers)               │
│ d_model=256, hidden=256  │
└──────────────────────────┘
        ↓
   Bahdanau (Additive) Attention
        ↓
┌──────────────────────────┐
│   DECODER                │
│ Unidirectional LSTM      │
│ (2 layers)               │
│ Attention context concat │
└──────────────────────────┘
        ↓
Output Tokens (coefficients separated by <BREAK>)
[SOS] + c0_tokens + [BREAK] + c1_tokens + [BREAK] + ... + [EOS]
```

**Architecture Details:**
- Embedding dimension: 256, LSTM hidden size: 256
- Encoder: 2-layer bidirectional LSTM
- Decoder: 2-layer unidirectional LSTM with Bahdanau attention
- Attention computes context vector from encoder outputs at each decoder step
- Weight tying: output projection shares weights with embedding
- Dropout: 0.1

**Shared across both models:**
- Same vocabulary (42 tokens) and special token handling
- Same public interface: `forward()`, `generate_batch()`, `generate()`
- Same checkpoint format (with added `model_type` key)

---

## 🏋️ Training Details

### Dataset

- **Source**: Taylor coefficient dataset (JSON format)
- **Format**: Each sample contains a function and its 5 Taylor coefficients
- **Train/Val Split**: 90% / 10%
- **Preprocessing**: Automatically filters out sequences exceeding `max_seq_len=512`

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 64 | Samples per batch |
| Learning Rate | 3e-4 | Initial Adam LR |
| Optimizer | Adam | Adam with default betas |
| Scheduler | Cosine Annealing | η_min = LR × 1e-2 |
| Epochs | 100 | Maximum training epochs |
| Gradient Clipping | 1.0 | Prevents exploding gradients |
| Loss Function | CrossEntropyLoss | Ignores PAD tokens |
| Max Seq Length | 512 | Truncates longer sequences |

### Training Loop

```python
For each epoch:
  1. Train on full training set (teacher forcing)
  2. Validate on validation set (teacher forcing - fast)
  3. Save checkpoint for every epoch
  4. Every 5 epochs: Evaluate on fixed test functions (autoregressive)
  5. Save best checkpoint by validation loss
```

### Loss Computation

- **Ignore index**: `PAD_ID` (padding tokens excluded from loss)
- **Reduction**: Mean across batch
- **Masking**: Automatic via `ignore_index` in CrossEntropyLoss

---

## 📈 Evaluation Metrics

### 1. **Token Accuracy**
```
Fraction of non-PAD tokens predicted correctly
Range: [0, 1] | 1.0 = perfect token-level predictions
```

### 2. **Sequence (Sentence) Accuracy**
```
Fraction of sequences where ALL tokens match ground truth
Range: [0, 1] | 1.0 = all sequences perfectly predicted
```

### 3. **Expression Validity**
```
Fraction of predicted sequences that form valid prefix expressions
(can be converted to infix without parse errors)
Range: [0, 1] | Catches structural/syntactic errors
```

### 4. **Per-Coefficient Accuracy**
```
Evaluated separately for each of the 5 coefficients
Split on <BREAK> tokens and compared individually
```

### 5. **Function-Level Accuracy**
```
Percentage of test functions where ALL 5 coefficients are correct
Shows if the model can jointly predict all coefficients accurately
```

### Evaluation Protocol

**During Training (Fast validation):**
- Uses teacher-forced decoding
- Measures on a batch sample from validation set

**Post-Training (Thorough evaluation):**
- Fixed set of 65 mathematical functions (e.g., `(x²+1)·sin(x)`, `exp(x)·log(1+x)`)
- Autoregressive greedy decoding (no teacher forcing)
- Per-coefficient and per-function accuracy reported

---

## 🚀 Running the Training

### Basic Training

```bash
python main.py
```

This will:
1. Load the dataset from the configured path
2. Build encoder-decoder model
3. Train for 100 epochs with validation
4. Save checkpoints to `checkpoints/` directory
5. Every 5 epochs: Evaluate on fixed test functions
6. At the end: Load best checkpoint and show example predictions

### Output

The training script prints:
- Dataset statistics (train/val split, samples skipped)
- Model size and vocabulary info
- Per-epoch: training loss, validation loss, elapsed time, best flag
- Every 5 epochs: Detailed evaluation on test functions
- Final: Best validation loss and example predictions

### Checkpoints

- **`checkpoints/epoch_NNN.pt`**: Checkpoint for every epoch
- **`checkpoints/taylor_series_pred_baseline.pt`**: Best checkpoint by validation loss

Each checkpoint contains:
```python
{
    "epoch": int,
    "model_state": state_dict,
    "val_loss": float,
    "n_coeffs": 5,
    "config": {architecture config dict}
}
```

---

## 🔍 Inference & Evaluation

### Using the Model for Inference

```python
import torch
from dataset import SOS_ID, EOS_ID, encode, decode, PAD_ID
from model import CoeffPredTransformer

# Load checkpoint
ckpt = torch.load("checkpoints/taylor_series_pred_baseline.pt", map_location="cpu")
model = CoeffPredTransformer(**ckpt["config"])
model.load_state_dict(ckpt["model_state"])
model.eval()

# Prepare input
fn_prefix_tokens = ["sin", "x"]  # Example
src_ids = [SOS_ID] + encode(fn_prefix_tokens) + [EOS_ID]
src = torch.tensor([src_ids], dtype=torch.long)

# Generate predictions (autoregressive)
with torch.no_grad():
    pred_ids = model.generate(src, max_len=512)

# Decode output
pred_tokens = decode(pred_ids)
print(f"Predicted tokens: {pred_tokens}")
```

### Evaluation Script

See `main.py` for:
- Fixed function list evaluation (every 5 epochs)
- Example predictions on validation set
- Function-level accuracy computation

---

## 📁 Project Structure

```
taylor-series-pred/
├── main.py                          # Training entry point
├── model.py                         # Transformer model architecture
├── dataset.py                       # Dataset class & vocabulary
├── dataset_generation.py            # Utilities for generating datasets
├── train_validate.py                # Training loop & validation
├── metrics.py                       # Evaluation metrics
├── inference.py                     # Inference utilities
├── checkpoints/                     # Saved model checkpoints
│   ├── epoch_001.pt
│   ├── epoch_002.pt
│   └── taylor_series_pred_baseline.pt  # Best model
├── train-taylor-series-task.ipynb   # Training notebook
├── demo-infer-nb.ipynb              # Inference demo notebook
└── dataset-gen-notebook.ipynb       # Dataset generation notebook
```

---

## 🧪 Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Complete training pipeline with evaluation |
| `model.py` | CoeffPredTransformer & CoeffPredLSTM (Seq2Seq architectures) |
| `dataset.py` | Vocabulary, encoding/decoding, dataset loader |
| `dataset_generation.py` | Tools for generating synthetic coefficient data |
| `train_validate.py` | Training loops, validation, optimizer setup |
| `metrics.py` | Token accuracy, sentence accuracy, validity checks |
| `inference.py` | Helper functions for model inference |
| `*.ipynb` | Jupyter notebooks for experimentation |

---

## 🎓 Example: Training from Scratch

```bash
# 1. Navigate to project
cd /path/to/taylor-series-pred

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Start training (adjust hyperparameters in main.py if needed)
python main.py

# 4. Monitor progress
# - Training/validation loss printed each epoch
# - Checkpoints saved to checkpoints/
# - Every 5 epochs: evaluation on test functions

# 5. After training, best model is at:
# checkpoints/taylor_series_pred_baseline.pt
```

---

## 🔧 Customization

Edit these variables in `main.py` to customize training:

```python
# Model selection
MODEL_TYPE = "transformer"  # "transformer" or "lstm"

# Transformer config (used when MODEL_TYPE = "transformer")
TRANSFORMER_CONFIG = {
    "d_model": 256, "nhead": 8,
    "num_encoder_layers": 6, "num_decoder_layers": 8,
    "dim_feedforward": 256, "dropout": 0.1, "max_seq_len": 512,
}

# LSTM config (used when MODEL_TYPE = "lstm")
LSTM_CONFIG = {
    "d_model": 256, "hidden_size": 256,
    "num_encoder_layers": 2, "num_decoder_layers": 2,
    "dropout": 0.1, "max_seq_len": 512,
}

# Data
DATASET_JSON = "path/to/taylor_dataset_10k.json"
VAL_RATIO = 0.10

# Training hyperparameters (shared)
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 3e-4
CLIP_GRAD = 1.0

# Evaluation
MAX_GEN_LEN = 512  # Max tokens to generate during inference
```


---



## 📝 Notes

- The model uses **prefix notation** (Polish notation) for all expressions
- **<BREAK> tokens** are critical for separating coefficients during decoding
- **Teacher forcing** used during training for speed; autoregressive decoding used for evaluation
- Dataset filtering removes sequences > 512 tokens
- Best checkpoint selected by validation loss, not test set accuracy

