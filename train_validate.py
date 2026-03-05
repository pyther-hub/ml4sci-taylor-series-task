"""
Taylor Series Dataset Generator — Single Config, Optimised
===========================================================
Based on Lample & Charton (2019) "Deep Learning for Symbolic Mathematics".

Key design choices:
  • No tiers — single flat Config (internal_nodes ∈ [2,6], max_depth ≤ 4)
  • Rational p/q → ['/', digit_tokens(p)..., digit_tokens(q)...]
      e.g.  1/3  →  ['/', '+', '1', '+', '3']
    'Rational' and '<INT>' tokens removed from vocabulary entirely.
  • Signal-based timeout (Unix/macOS) wraps any slow sympy call.
    If a call exceeds SIMPLIFY_TIMEOUT seconds the expression is skipped.
  • VERBOSE = True  prints per-sample stage timings to stdout.
  • SAVE_EVERY: JSON is overwritten after every N newly generated samples.
  • Config.x_bias controls how much more likely 'x' is than any single
    integer constant during leaf sampling.
      x_bias = 1.0  →  uniform  (x has same weight as each of -5…-1,1…5)
      x_bias = 5.0  →  x is 5× more likely than any individual constant
      x_bias = 10.0 →  x dominates almost all leaf slots
  • All user-facing config lives at the bottom in __main__.
"""

import sympy as sp
import random
import json
import os
import time
import signal
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# 0.  MODULE-LEVEL GLOBALS  (overridden in __main__)
# ══════════════════════════════════════════════════════════════════════════════

VERBOSE:          bool  = False   # per-sample stage timing output
SAVE_EVERY:       int   = 100     # overwrite JSON every N new samples
SIMPLIFY_TIMEOUT: int   = 10      # seconds before abandoning a sympy call


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TIMEOUT UTILITY  (SIGALRM — Unix / macOS only)
# ══════════════════════════════════════════════════════════════════════════════

class _Timeout(Exception):
    pass


def _sigalrm_handler(signum, frame):
    raise _Timeout()


def run_with_timeout(func, args=(), kwargs=None, seconds: int = 10, fallback=None):
    """
    Call func(*args, **kwargs).  If it doesn't return within `seconds`,
    cancel and return `fallback`.  Any other exception also returns `fallback`.

    Requires POSIX (Linux / macOS).  On Windows the alarm is skipped and the
    function runs without a time limit.
    """
    if kwargs is None:
        kwargs = {}

    # ── POSIX path ────────────────────────────────────────────────────────────
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.alarm(max(1, seconds))
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)
            return result
        except _Timeout:
            return fallback
        except Exception:
            return fallback
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    # ── Non-POSIX fallback (no timeout) ───────────────────────────────────────
    else:
        try:
            return func(*args, **kwargs)
        except Exception:
            return fallback


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════

VOCAB: list[str] = [
    # special tokens
    "<PAD>", "<SOS>", "<EOS>", "<UNK>",
    # variable and expansion point
    "x", "a",
    # sign tokens (digit-level number encoding)
    "+", "-",
    # digit characters
    *[str(i) for i in range(10)],
    # binary operators  ('+' and '-' are reused as sign tokens above)
    "*", "/", "**",
    # p1 = 15 unary functions
    "sin",   "cos",   "tan",
    "exp",   "log",   "sqrt",
    "asin",  "acos",  "atan",
    "sinh",  "cosh",  "tanh",
    "asinh", "acosh", "atanh",
    # parentheses (kept for potential infix use)
    "(", ")",
]

VOCAB_INDEX: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}

# Paper-exact operator / leaf sets
LEAVES:     list[str] = ["x"] + [str(i) for i in range(-5, 6) if i != 0]
BINARY_OPS: list[str] = ["+", "-", "*", "/"]
UNARY_OPS:  list[str] = [
    "sin",   "cos",   "tan",
    "exp",   "log",   "sqrt",
    "asin",  "acos",  "atan",
    "sinh",  "cosh",  "tanh",
    "asinh", "acosh", "atanh",
]

assert len(LEAVES)     == 11, "L must equal 11"
assert len(UNARY_OPS)  == 15, "p1 must equal 15"
assert len(BINARY_OPS) ==  4, "p2 must equal 4"

# SymPy symbols
x_sym = sp.Symbol("x")
a_sym = sp.Symbol("a")

UNARY_MAP: dict = {
    "sin":   sp.sin,   "cos":   sp.cos,   "tan":   sp.tan,
    "exp":   sp.exp,   "log":   sp.log,   "sqrt":  sp.sqrt,
    "asin":  sp.asin,  "acos":  sp.acos,  "atan":  sp.atan,
    "sinh":  sp.sinh,  "cosh":  sp.cosh,  "tanh":  sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
}
BINARY_MAP: dict = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  lambda a, b: a / b,
    "**": lambda a, b: a ** b,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DIGIT-LEVEL INTEGER TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════

def int_to_digit_tokens(n: int) -> list[str]:
    """
    Encode integer n as [sign, d1, d2, ...].
      0    → ['+', '0']
      42   → ['+', '4', '2']
     -123  → ['-', '1', '2', '3']
    """
    sign   = "-" if n < 0 else "+"
    digits = list(str(abs(n)))
    return [sign] + digits


def digit_tokens_to_int(tokens: list[str]) -> int:
    """Inverse of int_to_digit_tokens.  tokens = [sign, d1, d2, ...]"""
    sign   = -1 if tokens[0] == "-" else 1
    number = int("".join(tokens[1:]))
    return sign * number


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Single generation config — no tiers."""
    n_ops_range:  tuple = (2, 6)   # internal_nodes ∈ [2, 6]
    max_depth:    int   = 4
    max_nodes:    int   = 15       # generous upper bound for 6-op trees
    taylor_order: int   = 4
    x_bias:       float = 8.0      # weight of 'x' relative to each integer constant
                                   # during leaf sampling.
                                   #   1.0  → uniform  (x ≈ 9% chance, same as each const)
                                   #   5.0  → x is 5× more likely than any single const
                                   #          → x chosen ~31% of the time
                                   #  10.0  → x chosen ~52% of the time

    def leaf_weights(self) -> list[float]:
        """
        Return a weight list aligned with LEAVES = ['x', '-5', ..., '-1', '1', ..., '5'].
        The first entry is x_bias; all others are 1.0.
        """
        return [self.x_bias] + [1.0] * (len(LEAVES) - 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ALGORITHM 2 — uniform unary/binary tree sampling  (Lample & Charton)
# ══════════════════════════════════════════════════════════════════════════════

_D_CACHE: dict[tuple[int, int], int] = {}


def D(e: int, n: int) -> int:
    """Number of binary trees with e empty slots and n internal nodes."""
    if e == 0:
        return 0 if n > 0 else 1
    if n == 0:
        return 1
    key = (e, n)
    if key not in _D_CACHE:
        _D_CACHE[key] = D(e - 1, n) + D(e, n - 1) + D(e + 1, n - 1)
    return _D_CACHE[key]


def sample_tree(n_ops: int, max_depth: int, max_nodes: int) -> list:
    """Return a node-type list for a random expression tree."""
    nodes: list = []
    slots: list = [0]       # stack of current depths for open leaves
    remaining   = n_ops

    while remaining > 0 and slots:
        if len(nodes) >= max_nodes:
            break
        e = len(slots)
        if D(e, remaining) == 0:
            break

        choices, weights = [], []
        for k in range(e):
            if slots[k] < max_depth:
                w_u = D(e - k, remaining - 1)
                if w_u > 0:
                    choices.append(("unary",  k)); weights.append(w_u)
                w_b = D(e - k + 1, remaining - 1)
                if w_b > 0:
                    choices.append(("binary", k)); weights.append(w_b)
        if not choices:
            break

        arity_str, k = random.choices(choices, weights=weights, k=1)[0]
        parent_depth  = slots[k]

        # fill skipped slots with leaves
        for _ in range(k):
            nodes.append(("leaf", 0))
        slots = slots[k + 1:]

        if arity_str == "unary":
            nodes.append(("unary", 1))
            slots.insert(0, parent_depth + 1)
        else:
            nodes.append(("binary", 2))
            slots = [parent_depth + 1, parent_depth + 1] + slots

        remaining -= 1

    # close any remaining open slots with leaves
    for _ in slots:
        if len(nodes) < max_nodes:
            nodes.append(("leaf", 0))

    return nodes


def _build_expr(nodes: list, leaf_weights: list[float]) -> tuple[sp.Expr, list]:
    """
    Recursively consume the node list and produce a sympy expression.

    `leaf_weights` is a list aligned with LEAVES (length 11).
    The first entry controls how often 'x' is chosen relative to integer
    constants.  Produced by Config.leaf_weights().
    """
    if not nodes:
        return sp.Integer(1), []

    node_type, _ = nodes[0]
    rest = nodes[1:]

    if node_type == "leaf":
        leaf = random.choices(LEAVES, weights=leaf_weights, k=1)[0]
        return (x_sym if leaf == "x" else sp.Integer(int(leaf))), rest

    if node_type == "unary":
        op           = random.choice(UNARY_OPS)
        child, rest  = _build_expr(rest, leaf_weights)
        try:
            return UNARY_MAP[op](child), rest
        except Exception:
            return child, rest

    # binary
    op            = random.choice(BINARY_OPS)
    left,  rest   = _build_expr(rest, leaf_weights)
    right, rest   = _build_expr(rest, leaf_weights)
    try:
        return BINARY_MAP[op](left, right), rest
    except Exception:
        return left, rest


# ── Cheap structural metrics (no heavy sympy) ─────────────────────────────────

def _depth(expr: sp.Expr) -> int:
    if expr.is_Atom:
        return 0
    return 1 + max(_depth(arg) for arg in expr.args)


def _node_count(expr: sp.Expr) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


# ── Full random-function generator with guards ────────────────────────────────

def generate_random_function(cfg: Config, retries: int = 25) -> sp.Expr | None:
    """
    Sample a random function, apply guards, and return a valid sympy expr.
    Returns None if every attempt fails or times out.

    Leaf sampling is biased by cfg.x_bias so that 'x' appears more
    frequently than integer constants.
    """
    leaf_weights = cfg.leaf_weights()   # precompute once per call

    for _ in range(retries):
        n_ops  = random.randint(*cfg.n_ops_range)
        nodes  = sample_tree(n_ops, cfg.max_depth, cfg.max_nodes)
        raw, _ = _build_expr(nodes, leaf_weights)

        # cheap guards (no sympy simplification yet)
        if not raw.has(x_sym) or raw.is_number:
            continue
        if _depth(raw) > cfg.max_depth or _node_count(raw) > cfg.max_nodes:
            continue

        # timed nsimplify
        expr = run_with_timeout(
            sp.nsimplify, args=(raw,), kwargs={"rational": True},
            seconds=SIMPLIFY_TIMEOUT, fallback=None,
        )
        if expr is None:
            continue

        if not expr.has(x_sym) or expr.is_number:
            continue
        if _depth(expr) > cfg.max_depth or _node_count(expr) > cfg.max_nodes:
            continue

        try:
            if sp.diff(expr, x_sym) == 0:
                continue
        except Exception:
            continue

        return expr

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TAYLOR SERIES COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_taylor_coefficients(
    expr: sp.Expr, order: int
) -> list[sp.Expr] | None:
    """
    Return [c0, c1, ..., c_order]  where  c_n = d^n f / dx^n  evaluated at x=a.
    (Coefficients are NOT divided by n!.)

    Uses sp.cancel() for fast rational simplification.
    Each call is wrapped in a timeout; returns None if any coefficient times out.
    """
    coeffs: list[sp.Expr] = []
    deriv = expr
    for n in range(order + 1):
        if n > 0:
            deriv = sp.diff(deriv, x_sym)
        substituted = deriv.subs(x_sym, a_sym)
        result = run_with_timeout(
            sp.cancel, args=(substituted,),
            seconds=SIMPLIFY_TIMEOUT, fallback=None,
        )
        if result is None:
            return None
        coeffs.append(result)
    return coeffs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PREFIX TOKENISATION  (digit-level; Rational → '/' operator)
# ══════════════════════════════════════════════════════════════════════════════

def _expr_to_prefix_tokens(expr: sp.Expr) -> list[str]:
    """
    Convert sympy expression to a prefix-notation token list.

    Encoding rules
    ──────────────
    • Integer n          →  int_to_digit_tokens(n)
                             e.g.  -3  →  ['-', '3']
    • Rational p/q       →  ['/', *int_to_digit_tokens(p), *int_to_digit_tokens(q)]
                             e.g.  1/3 →  ['/', '+', '1', '+', '3']
    • Binary op          →  [op, left_tokens..., right_tokens...]
    • Unary fn           →  [fname, child_tokens...]
    • x, a               →  ['x'], ['a']

    Result is wrapped with <SOS> and <EOS>.
    """
    def _rec(e) -> list[str]:
        # ── Numbers ──────────────────────────────────────────────────────────
        if e.is_Integer:
            return int_to_digit_tokens(int(e))

        if e.is_Rational:          # non-integer rational  p/q
            return ["/"] + int_to_digit_tokens(int(e.p)) + int_to_digit_tokens(int(e.q))

        if e.is_Number:            # Float / other (shouldn't normally occur)
            return [str(e)]

        # ── Symbols ───────────────────────────────────────────────────────────
        if e == x_sym: return ["x"]
        if e == a_sym: return ["a"]

        # ── Addition ─────────────────────────────────────────────────────────
        if e.is_Add:
            args = list(e.args)
            t = _rec(args[0])
            for arg in args[1:]:
                t = ["+"] + t + _rec(arg)
            return t

        # ── Multiplication ────────────────────────────────────────────────────
        if e.is_Mul:
            args = list(e.args)
            t = _rec(args[0])
            for arg in args[1:]:
                t = ["*"] + t + _rec(arg)
            return t

        # ── Power ─────────────────────────────────────────────────────────────
        if e.is_Pow:
            base, exp_ = e.args
            return ["**"] + _rec(base) + _rec(exp_)

        # ── Unary functions ───────────────────────────────────────────────────
        fname = type(e).__name__.lower()
        if fname in VOCAB_INDEX:
            return [fname] + _rec(e.args[0])

        # ── Fallback ──────────────────────────────────────────────────────────
        return [str(e)]

    try:
        tokens = _rec(sp.nsimplify(expr, rational=True))
    except Exception:
        tokens = [str(expr)]

    return  tokens 


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PREFIX → INFIX RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _consume_number(tokens: list[str], pos: int) -> tuple[sp.Expr, int]:
    """
    Consume a digit-level integer starting at pos (expects sign token first).
    Returns (sympy Integer, new_pos).
    """
    sign = tokens[pos]; pos += 1
    digits: list[str] = []
    while pos < len(tokens) and tokens[pos].isdigit():
        digits.append(tokens[pos]); pos += 1
    value = int("".join(digits)) * (-1 if sign == "-" else 1)
    return sp.Integer(value), pos


def _prefix_tokens_to_expr(tokens: list[str], pos: int = 0) -> tuple[sp.Expr, int]:
    """Parse prefix token list (no <SOS>/<EOS>) back to a sympy expression."""
    if pos >= len(tokens):
        return sp.Integer(1), pos

    tok = tokens[pos]

    # ── Digit-level integer  (sign token followed by digits) ─────────────────
    if tok in ("+", "-") and pos + 1 < len(tokens) and tokens[pos + 1].isdigit():
        return _consume_number(tokens, pos)

    # ── Leaf symbols ──────────────────────────────────────────────────────────
    if tok == "x": return x_sym, pos + 1
    if tok == "a": return a_sym, pos + 1

    # ── Binary operators ──────────────────────────────────────────────────────
    if tok in ("+", "-", "*", "/", "**"):
        left,  pos = _prefix_tokens_to_expr(tokens, pos + 1)
        right, pos = _prefix_tokens_to_expr(tokens, pos)
        return BINARY_MAP[tok](left, right), pos

    # ── Unary functions ───────────────────────────────────────────────────────
    if tok in UNARY_MAP:
        child, pos = _prefix_tokens_to_expr(tokens, pos + 1)
        try:
            return UNARY_MAP[tok](child), pos
        except Exception:
            return child, pos

    # ── Fallback: unknown token treated as symbol ─────────────────────────────
    return sp.Symbol(tok), pos + 1


def prefix_tokens_to_infix(tokens: list[str]) -> str:
    """
    Reconstruct an infix string from a prefix token list.
    Strips <SOS>, <EOS>, <PAD> before parsing.
    """
    clean = [t for t in tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
    try:
        expr, _ = _prefix_tokens_to_expr(clean, 0)
        return str(expr)
    except Exception as exc:
        return f"<PARSE_ERROR: {exc}>"


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _save_json(data: list, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def generate_dataset(
    cfg:         Config,
    n_samples:   int,
    seed:        int,
    output_path: str,
) -> list[dict]:
    """
    Generate `n_samples` valid (function, Taylor-coefficients) pairs.

    Progress
    ────────
    • Prints a one-line summary per sample when VERBOSE = True.
    • Saves (overwrites) `output_path` every SAVE_EVERY new samples.
    • Always saves at the end.

    Returns the complete list of sample dicts.
    """
    random.seed(seed)

    dataset:  list[dict] = []
    attempts: int        = 0
    t_wall                = time.perf_counter()

    # per-stage accumulators for end-of-run summary
    _t: dict[str, list[float]] = {
        "generate_fn":    [],
        "taylor_coeffs":  [],
        "prefix_tokens":  [],
    }
    while len(dataset) < n_samples and attempts < n_samples * 100:
        try:
                attempts += 1
        
                # ── Stage: generate function ─────────────────────────────────────────
                t0   = time.perf_counter()
                expr = generate_random_function(cfg)
                t_gen = time.perf_counter() - t0
                if expr is None:
                    continue
                _t["generate_fn"].append(t_gen)
        
                # ── Stage: Taylor coefficients ───────────────────────────────────────
                t0     = time.perf_counter()
                coeffs = compute_taylor_coefficients(expr, cfg.taylor_order)
                t_tay  = time.perf_counter() - t0
                if coeffs is None:
                    continue
                if any(c.has(sp.zoo, sp.nan, sp.oo, sp.I) for c in coeffs):
                    continue
                _t["taylor_coeffs"].append(t_tay)
        
                # ── Stage: prefix tokenisation ───────────────────────────────────────
                t0          = time.perf_counter()
                fn_tokens   = _expr_to_prefix_tokens(expr)
                c_tokens    = [_expr_to_prefix_tokens(c) for c in coeffs]
                t_tok       = time.perf_counter() - t0
                _t["prefix_tokens"].append(t_tok)
        
                # ── Assemble sample ──────────────────────────────────────────────────
                depth  = _depth(expr)
                nodes  = _node_count(expr)
                n_ops  = sum(1 for nd in sp.preorder_traversal(expr) if not nd.is_Atom)
        
                coeff_entries = {
                    f"coeff{n}": {"infix": str(cn), "prefix": c_tokens[n]}
                    for n, cn in enumerate(coeffs)
                }
        
                sample = {
                    "id": f"sample_{len(dataset):05d}",
                    "function": {
                        "infix":  str(expr),
                        "prefix": fn_tokens,
                    },
                    "taylor_series": {
                        "coefficients": coeff_entries,
                    },
                    "metadata": {
                        "n_ops":        n_ops,
                        "tree_depth":   depth,
                        "tree_nodes":   nodes,
                        "taylor_order": cfg.taylor_order,
                    },
                }
        except Exception as error:
            print("An exception occurred:", error)
            continue
            
        dataset.append(sample)
        count = len(dataset)

        # ── Verbose per-sample line ──────────────────────────────────────────
        if VERBOSE:
            elapsed = time.perf_counter() - t_wall
            print(
                f"  [{count:05d}/{n_samples}]  "
                f"depth={depth}  nodes={nodes:2d}  ops={n_ops}  "
                f"gen={t_gen*1e3:6.1f}ms  "
                f"taylor={t_tay*1e3:6.1f}ms  "
                f"tok={t_tok*1e3:5.2f}ms  "
                f"wall={elapsed:.1f}s"
            )
            if (t_tay*1e3>3000):
                print(str(expr))
        

        # ── Periodic checkpoint save ─────────────────────────────────────────
        if count % SAVE_EVERY == 0:
            _save_json(dataset, output_path)
            if not VERBOSE:
                print(f"  [checkpoint] {count}/{n_samples} samples  →  {output_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    _save_json(dataset, output_path)

    yield_pct = 100 * len(dataset) / attempts if attempts else 0.0
    total_s   = time.perf_counter() - t_wall

    print(f"\n  Generated : {len(dataset)}/{n_samples} samples")
    print(f"  Attempts  : {attempts}  (yield {yield_pct:.1f}%)")
    print(f"  Wall time : {total_s:.2f}s")

    if _t["generate_fn"]:
        print(f"\n  Stage averages (over successful attempts):")
        for stage, times in _t.items():
            if times:
                avg_ms = sum(times) / len(times) * 1000
                print(f"    {stage:<18}  {avg_ms:7.3f} ms  (n={len(times)})")

    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ROUNDTRIP TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_roundtrip(cfg: Config, n: int = 10, seed: int = 99) -> None:
    """
    Generate `n` expressions, encode to prefix, decode back to infix, and
    verify symbolic equality.  Also validates Taylor coefficient encoding.
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  TEST: prefix ↔ infix roundtrip  (digit-level, Rational→'/')")
    print(SEP)

    random.seed(seed)
    passed = failed = tested = 0

    while tested < n:
        expr = generate_random_function(cfg)
        if expr is None:
            continue

        # ── Function roundtrip ────────────────────────────────────────────────
        tokens        = _expr_to_prefix_tokens(expr)
        reconstructed = prefix_tokens_to_infix(tokens)
        try:
            recon_expr = sp.sympify(reconstructed, locals={"x": x_sym, "a": a_sym})
            ok_func    = sp.simplify(expr - recon_expr) == 0
        except Exception:
            ok_func    = False

        # ── Coefficient roundtrip ─────────────────────────────────────────────
        coeffs    = compute_taylor_coefficients(expr, order=4)
        ok_coeffs = True
        if coeffs:
            for n_idx, cn in enumerate(coeffs):
                if cn.has(sp.zoo, sp.nan, sp.oo, sp.I):
                    continue
                c_toks   = _expr_to_prefix_tokens(cn)
                c_recon  = prefix_tokens_to_infix(c_toks)
                try:
                    c_expr   = sp.sympify(c_recon, locals={"x": x_sym, "a": a_sym})
                    c_ok     = sp.simplify(cn - c_expr) == 0
                except Exception:
                    c_ok = False
                if not c_ok:
                    ok_coeffs = False
                    print(f"    [COEFF FAIL] coeff{n_idx}: {cn}  →  tokens={c_toks}  →  {c_recon}")
                    break

        tested += 1
        if ok_func and ok_coeffs:
            passed += 1
            print(f"  [PASS #{tested:02d}]  f(x) = {expr}")
            print(f"            prefix = {tokens}")
            print(f"            recon  = {reconstructed}")
        else:
            failed += 1
            diff = sp.simplify(expr - recon_expr) if ok_func is False else "coeff mismatch"
            print(f"  [FAIL #{tested:02d}]  f(x)={expr}  diff={diff}")

    print("-" * 65)
    print(f"  Result: {passed}/{tested} passed,  {failed} failed")
    print("  ALL ROUNDTRIP TESTS PASSED ✓" if failed == 0 else "  SOME TESTS FAILED ✗")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(cfg: Config, n_trials: int = 50, seed: int = 123) -> None:
    """
    Time each pipeline stage independently over `n_trials` completed samples.

    Stages:
      1  sample_tree           — Algorithm 2 node-list
      2  _build_expr           — node-list → raw sympy
      3  nsimplify             — rational simplification
      4  taylor_coefficients   — symbolic differentiation × (order+1)
      5  expr_prefix_tokens    — f(x) → token list
      6  coeff_prefix_tokens   — per-coefficient encoding (averaged)
      7  prefix_to_infix       — token list → infix string
      8  full_build_sample     — end-to-end build_sample()
    """
    random.seed(seed)
    stages = [
        "1_sample_tree",        "2_build_expr",
        "3_nsimplify",          "4_taylor_coefficients",
        "5_expr_prefix_tokens", "6_coeff_prefix_tokens",
        "7_prefix_to_infix",    "8_full_build_sample",
    ]
    timings: dict[str, list[float]] = {s: [] for s in stages}

    completed  = attempts = 0
    leaf_weights = cfg.leaf_weights()   # precompute once for the whole benchmark

    while completed < n_trials and attempts < n_trials * 100:
        try:
            attempts += 1
            n_ops = random.randint(*cfg.n_ops_range)

            # 1 ── sample_tree
            t0 = time.perf_counter()
            nodes = sample_tree(n_ops, cfg.max_depth, cfg.max_nodes)
            timings["1_sample_tree"].append(time.perf_counter() - t0)

            # 2 ── _build_expr
            t0 = time.perf_counter()
            raw_expr, _ = _build_expr(nodes, leaf_weights)
            timings["2_build_expr"].append(time.perf_counter() - t0)

            # 3 ── nsimplify (timed)
            t0 = time.perf_counter()
            expr = run_with_timeout(
                sp.nsimplify, args=(raw_expr,), kwargs={"rational": True},
                seconds=SIMPLIFY_TIMEOUT, fallback=None,
            )
            timings["3_nsimplify"].append(time.perf_counter() - t0)
            if expr is None:
                continue

            if not expr.has(x_sym) or expr.is_number:
                continue
            if _depth(expr) > cfg.max_depth or _node_count(expr) > cfg.max_nodes:
                continue
            try:
                if sp.diff(expr, x_sym) == 0:
                    continue
            except Exception:
                continue

            # 4 ── Taylor coefficients
            t0     = time.perf_counter()
            coeffs = compute_taylor_coefficients(expr, cfg.taylor_order)
            timings["4_taylor_coefficients"].append(time.perf_counter() - t0)
            if coeffs is None or any(c.has(sp.zoo, sp.nan, sp.oo, sp.I) for c in coeffs):
                continue

            # 5 ── expr → prefix tokens
            t0          = time.perf_counter()
            expr_tokens = _expr_to_prefix_tokens(expr)
            timings["5_expr_prefix_tokens"].append(time.perf_counter() - t0)

            # 6 ── coeff → prefix tokens (per-coeff average)
            t0 = time.perf_counter()
            for cn in coeffs:
                _expr_to_prefix_tokens(cn)
            timings["6_coeff_prefix_tokens"].append(
                (time.perf_counter() - t0) / len(coeffs)
            )

            # 7 ── prefix → infix
            t0 = time.perf_counter()
            prefix_tokens_to_infix(expr_tokens)
            timings["7_prefix_to_infix"].append(time.perf_counter() - t0)

            # 8 ── full build_sample
            t0     = time.perf_counter()
            sample = _build_sample_internal(completed, cfg)
            t_full = time.perf_counter() - t0
            if sample is None:
                continue
            timings["8_full_build_sample"].append(t_full)

            completed += 1

        except Exception as error:
            print(error)

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  BENCHMARK  n_trials={n_trials}  completed={completed}  "
          f"yield={100*completed/attempts:.1f}%")
    print(f"  Config: ops={cfg.n_ops_range}  depth≤{cfg.max_depth}  "
          f"nodes≤{cfg.max_nodes}  order={cfg.taylor_order}")
    print(SEP)
    print(f"  {'Stage':<32}  {'n':>4}  {'mean':>9}  {'min':>9}  {'max':>9}  {'total':>10}")
    print(f"  {'-'*32}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")
    for stage, times in timings.items():
        if not times:
            print(f"  {stage:<32}  {'—':>4}")
            continue
        n   = len(times)
        ms  = lambda v: v * 1000
        print(
            f"  {stage:<32}  {n:>4}  "
            f"{ms(sum(times)/n):>8.3f}ms  "
            f"{ms(min(times)):>8.3f}ms  "
            f"{ms(max(times)):>8.3f}ms  "
            f"{ms(sum(times)):>9.2f}ms"
        )
    print(SEP + "\n")


def _build_sample_internal(sample_id: int, cfg: Config) -> dict | None:
    """Thin wrapper used by the benchmark to time a full end-to-end sample."""
    expr = generate_random_function(cfg)
    if expr is None:
        return None
    coeffs = compute_taylor_coefficients(expr, cfg.taylor_order)
    if coeffs is None or any(c.has(sp.zoo, sp.nan, sp.oo, sp.I) for c in coeffs):
        return None
    n_ops = sum(1 for nd in sp.preorder_traversal(expr) if not nd.is_Atom)
    return {
        "id": f"sample_{sample_id:05d}",
        "function": {"infix": str(expr), "prefix": _expr_to_prefix_tokens(expr)},
        "taylor_series": {
            "coefficients": {
                f"coeff{n}": {"infix": str(cn), "prefix": _expr_to_prefix_tokens(cn)}
                for n, cn in enumerate(coeffs)
            }
        },
        "metadata": {
            "n_ops": n_ops,
            "tree_depth":   _depth(expr),
            "tree_nodes":   _node_count(expr),
            "taylor_order": cfg.taylor_order,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════

# ╔══════════════════════════════════════════════════════╗
# ║                  U S E R   C O N F I G              ║
# ╠══════════════════════════════════════════════════════╣
VERBOSE           = True          # True  → print per-sample timings
SAVE_EVERY        = 100           # overwrite JSON every N new samples
SIMPLIFY_TIMEOUT  = 10            # seconds before skipping a slow sympy call

N_SAMPLES         = 85000           # total samples to generate

SEED              = random.randint(0,1000)+64+84+454
OUTPUT_PATH       = "datasets/taylor_dataset.json"

cfg = Config(
    n_ops_range  = (2, 6),        # internal_nodes ∈ [2, 6]
    max_depth    = 4,
    max_nodes    = 14,
    taylor_order = 4,
    x_bias       = 14.0,           # how much more likely 'x' is vs any single
                                  # integer constant at each leaf slot.
                                  #   1.0  → uniform  (~9% x per leaf)
                                  #   5.0  → ~31% x per leaf  ← recommended
                                  #  10.0  → ~52% x per leaf
)
# ╚══════════════════════════════════════════════════════╝

print("Taylor Series Dataset Generator")
print(f"  Config : ops={cfg.n_ops_range}  depth≤{cfg.max_depth}  "
      f"nodes≤{cfg.max_nodes}  order={cfg.taylor_order}")
print(f"  Samples: {N_SAMPLES}   seed={SEED}   save_every={SAVE_EVERY}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Timeout: {SIMPLIFY_TIMEOUT}s per sympy call\n")

# ── 1. Roundtrip sanity test ─────────────────────────────────────────────
test_roundtrip(cfg=cfg, n=10, seed=SEED)

# ── 2. Generate dataset ──────────────────────────────────────────────────
t0      = time.perf_counter()
dataset = generate_dataset(
    cfg         = cfg,
    n_samples   = N_SAMPLES,
    seed        = SEED,
    output_path = OUTPUT_PATH,
)
print(f"\nFinal save → {OUTPUT_PATH}  ({len(dataset)} samples, "
      f"{time.perf_counter()-t0:.2f}s total)\n")

# ── 3. Benchmark ─────────────────────────────────────────────────────────
benchmark(cfg=cfg, n_trials=50, seed=SEED)


# ╔══════════════════════════════════════════════════════╗
# ║                  U S E R   C O N F I G              ║
# ╠══════════════════════════════════════════════════════╣
VERBOSE           = True          # True  → print per-sample timings
SAVE_EVERY        = 100           # overwrite JSON every N new samples
SIMPLIFY_TIMEOUT  = 10            # seconds before skipping a slow sympy call

N_SAMPLES         = 10000           # total samples to generate

SEED              = random.randint(0,100000)
OUTPUT_PATH       = "datasets/taylor_dataset.json"

cfg = Config(
    n_ops_range  = (2, 6),        # internal_nodes ∈ [2, 6]
    max_depth    = 4,
    max_nodes    = 15,
    taylor_order = 4,
    x_bias       = 5.0,           # how much more likely 'x' is vs any single
                                  # integer constant at each leaf slot.
                                  #   1.0  → uniform  (~9% x per leaf)
                                  #   5.0  → ~31% x per leaf  ← recommended
                                  #  10.0  → ~52% x per leaf
)
# ╚══════════════════════════════════════════════════════╝

print("Taylor Series Dataset Generator")
print(f"  Config : ops={cfg.n_ops_range}  depth≤{cfg.max_depth}  "
      f"nodes≤{cfg.max_nodes}  order={cfg.taylor_order}")
print(f"  Samples: {N_SAMPLES}   seed={SEED}   save_every={SAVE_EVERY}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Timeout: {SIMPLIFY_TIMEOUT}s per sympy call\n")

# ── 1. Roundtrip sanity test ─────────────────────────────────────────────
test_roundtrip(cfg=cfg, n=10, seed=SEED)

# ── 2. Generate dataset ──────────────────────────────────────────────────
t0      = time.perf_counter()
dataset = generate_dataset(
    cfg         = cfg,
    n_samples   = N_SAMPLES,
    seed        = SEED,
    output_path = OUTPUT_PATH,
)
print(f"\nFinal save → {OUTPUT_PATH}  ({len(dataset)} samples, "
      f"{time.perf_counter()-t0:.2f}s total)\n")

# ── 3. Benchmark ─────────────────────────────────────────────────────────
benchmark(cfg=cfg, n_trials=50, seed=SEED)