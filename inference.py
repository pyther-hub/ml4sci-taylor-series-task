import torch
import sympy as sp

from dataset import EOS_ID, PAD_ID, SOS_ID, VOCAB_SIZE, encode, decode
from dataset_generation import _expr_to_prefix_tokens, prefix_tokens_to_infix
from model import CoeffPredLSTM, CoeffPredTransformer

MODEL_CLASSES = {
    "transformer": CoeffPredTransformer,
    "lstm":        CoeffPredLSTM,
}

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "models/v4_128_op_epoch_010.pt"
MAX_GEN_LEN        = 128


def load_model(checkpoint_path: str, device: torch.device):
    """Load model weights and architecture from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    model_type = ckpt.get("model_type", "transformer")  # backward compat

    cls   = MODEL_CLASSES[model_type]
    model = cls(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint : {checkpoint_path}")
    print(f"  Model type        : {model_type}")
    print(f"  Epoch             : {ckpt.get('epoch', '?')}")
    print(f"  Val loss          : {ckpt.get('val_loss', float('nan')):.6f}")
    print(f"  N coefficients    : {ckpt.get('n_coeffs', '?')}\n")
    return model

def predict(expression: str, model: CoeffPredTransformer, device: torch.device) -> str:
    """
    Tokenise a single expression string, run greedy decoding, and return
    the decoded prediction string.
    """
    # Parse expression string → sympy → prefix token list (matches training format)
    expr_sym = sp.sympify(expression)
    prefix_tokens = _expr_to_prefix_tokens(expr_sym)
    print(f"  Prefix tokens : {prefix_tokens}")

    # Encode with SOS/EOS wrapping, matching dataset.py src format:
    #   src = [SOS_ID] + encode(fn_tokens) + [EOS_ID]
    token_ids = [SOS_ID] + encode(prefix_tokens) + [EOS_ID]
    print(f"  Token IDs     : {token_ids}")
    src = torch.tensor([token_ids], dtype=torch.long).to(device)  # (1, L)

    with torch.no_grad():
        # generate_batch returns a list of token-id lists (one per batch item)
        pred_ids = model.generate_batch(src, max_len=MAX_GEN_LEN)[0]

    # Decode: skip_special=True removes PAD/SOS/EOS but keeps <BREAK>
    pred_tokens = decode(pred_ids, skip_special=True)
    pred_str = " ".join(pred_tokens)
    return pred_str


checkpoint =  "models/v4_128_op_epoch_010.pt"
expr = "x*(-sin(x) - 1 + 1/x)"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}\n")

# ── Load model ────────────────────────────────────────────────────────
model = load_model(checkpoint, device)


# ── Run inference ─────────────────────────────────────────────────────
print(f"  Input expression : {expr}")
prediction = predict(expr, model, device)
print(f"  Predicted coeffs : {prediction}")
print()

# ── Pretty-print individual coefficients (split on <BREAK> token) ─────
# Denominators for Taylor series: n! for n = 0,1,2,3,4
FACTORIALS = [1, 1, 2, 6, 24]

coeff_strs = [c.strip() for c in prediction.split("<BREAK>")]
print(f"  Individual coefficients ({len(coeff_strs)} found):")

infix_coeffs = []
for i, c_str in enumerate(coeff_strs):
    c_tokens = c_str.split()
    c_infix  = prefix_tokens_to_infix(c_tokens)
    denom    = FACTORIALS[i] if i < len(FACTORIALS) else "?"
    print(f"    c[{i}]  prefix : {c_str}")
    print(f"    c[{i}]  infix  : {c_infix}   [n!= {denom}]")
    infix_coeffs.append(c_infix)

# ── Complete Taylor series equation ────────────────────────────────────
print()
terms = []
for i, (c_infix, denom) in enumerate(zip(infix_coeffs, FACTORIALS)):
    coeff_part = f"({c_infix})/{denom}"
    if i == 0:
        terms.append(coeff_part)
    else:
        power_part = "(x-a)" if i == 1 else f"(x-a)^{i}"
        terms.append(f"{coeff_part}*{power_part}")

print(f"  Complete Taylor series:")
print(f"    f(x) ≈ {' + '.join(terms)}")
