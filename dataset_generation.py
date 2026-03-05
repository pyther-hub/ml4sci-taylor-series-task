"""
Taylor Series Dataset Generator — POC Edition
==============================================
Based on Lample & Charton (2019) "Deep Learning for Symbolic Mathematics".

POC design choices:
  • 2–4 internal nodes, max_depth = 3, max_nodes = 8
  • Binary ops  : +, -, *, /
  • Unary fns   : sin, cos, exp, log, sqrt
  • Power nodes : x**2 and x**3 ONLY  (base must be x, exponent 2 or 3)
  • Leaves      : x  |  constants from [-3,-2,-1,-1,1,1,1,2,3]
                  (1 appears 3× so it has higher probability)
  • No floats anywhere — integer constants only
  • Rational p/q → ['/', digit_tokens(p)..., digit_tokens(q)...]
  • Signal-based timeout (Unix/macOS) wraps any slow sympy call.
  • VERBOSE = True  prints per-sample stage timings to stdout.
  • SAVE_EVERY: JSON is overwritten after every N newly generated samples.
"""

import sympy as sp
import random
import json
import os
import time
import signal
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# 0.  MODULE-LEVEL GLOBALS
# ══════════════════════════════════════════════════════════════════════════════

VERBOSE:          bool = False
SAVE_EVERY:       int  = 100
SIMPLIFY_TIMEOUT: int  = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TIMEOUT UTILITY  (SIGALRM — Unix / macOS only)
# ══════════════════════════════════════════════════════════════════════════════

class _Timeout(Exception):
    pass


def _sigalrm_handler(signum, frame):
    raise _Timeout()


def run_with_timeout(func, args=(), kwargs=None, seconds: int = 10, fallback=None):
    if kwargs is None:
        kwargs = {}
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
    # binary operators
    "*", "/", "**",
    # unary functions (5 for POC)
    "sin", "cos", "exp", "log", "sqrt",
    # parentheses
    "(", ")",
]

VOCAB_INDEX: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}

# ── POC-specific operator / leaf sets ─────────────────────────────────────────

# Leaf pool: 1 appears 3× for higher priority; no floats
LEAF_POOL: list[str] = ["x", "-3", "-2", "-1", "-1", "1", "1", "1", "2", "3"]

# Integer constants available as leaves (unique values for the pool)
INT_CONSTANTS: list[int] = [-3, -2, -1, 1, 2, 3]

BINARY_OPS: list[str] = ["+", "-", "*", "/", "**"]
UNARY_OPS:  list[str] = ["sin", "cos", "exp", "log", "sqrt"]

# Valid exponents for power nodes (x**2 and x**3 only)
VALID_EXPONENTS: list[int] = [2, 3]

# SymPy symbols
x_sym = sp.Symbol("x")
a_sym = sp.Symbol("a")

UNARY_MAP: dict = {
    "sin":  sp.sin,
    "cos":  sp.cos,
    "exp":  sp.exp,
    "log":  sp.log,
    "sqrt": sp.sqrt,
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
    sign   = -1 if tokens[0] == "-" else 1
    number = int("".join(tokens[1:]))
    return sign * number


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """POC generation config — simple functions only."""
    n_ops_range:  tuple = (2, 5)   # internal_nodes ∈ [2, 5]
    max_depth:    int   = 4        # 4 needed: e.g. x/(1-x) has sympy depth 4
    max_nodes:    int   = 12       # 12 needed: e.g. (2x+1)*sin(3x) has 10 nodes
    taylor_order: int   = 4
    x_bias:       float = 4.0


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
    slots: list = [0]
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

    for _ in slots:
        if len(nodes) < max_nodes:
            nodes.append(("leaf", 0))

    return nodes


def _sample_leaf(force_x: bool = False) -> sp.Expr:
    """
    Sample a leaf.
    If force_x=True, always returns x_sym (used when we must guarantee x in a subtree).
    Otherwise samples from LEAF_POOL with extra x slots for bias.
    """
    if force_x:
        return x_sym
    pool   = LEAF_POOL + ["x"] * 3   # 'x' gets +3 extra slots for higher probability
    chosen = random.choice(pool)
    return x_sym if chosen == "x" else sp.Integer(int(chosen))


def _build_expr(nodes: list, must_contain_x: bool = False) -> tuple[sp.Expr, list, bool]:
    """
    Recursively consume the node list and produce a sympy expression.

    Returns (expr, remaining_nodes, contains_x).

    Key constraint (Paper Appendix C + POC rules):
      • A UNARY node's child MUST contain x.
        → If the child subtree has no x, we force a leaf 'x' before applying the fn.
      • A BINARY '**' node → always produces x**2 or x**3, always contains x.
      • A BINARY +/-/* / node → left and right built freely; if must_contain_x
        and neither child has x, the left child is rebuilt as x.
      • This ensures no expression like sin(3), sqrt(5), exp(-1) can appear.

    must_contain_x: when True, this subtree MUST include x somewhere.
    """
    if not nodes:
        # Fallback leaf
        leaf = _sample_leaf(force_x=must_contain_x)
        return leaf, [], leaf == x_sym

    node_type, _ = nodes[0]
    rest = nodes[1:]

    # ── Leaf ─────────────────────────────────────────────────────────────────
    if node_type == "leaf":
        leaf = _sample_leaf(force_x=must_contain_x)
        return leaf, rest, leaf == x_sym

    # ── Unary ────────────────────────────────────────────────────────────────
    if node_type == "unary":
        op = random.choice(UNARY_OPS)
        # Child MUST contain x — enforce it
        child, rest, child_has_x = _build_expr(rest, must_contain_x=True)
        if not child_has_x:
            # Safety fallback: replace with x directly
            child = x_sym
        try:
            result = UNARY_MAP[op](child)
        except Exception:
            result = child
        return result, rest, True   # always contains x since child does

    # ── Binary ───────────────────────────────────────────────────────────────
    op = random.choice(BINARY_OPS)

    # Power: always x**2 or x**3 — consume child slots but discard them
    if op == "**":
        exp_val    = random.choice(VALID_EXPONENTS)   # 2 or 3
        _, rest, _ = _build_expr(rest)                # consume left slot
        _, rest, _ = _build_expr(rest)                # consume right slot
        return x_sym ** sp.Integer(exp_val), rest, True

    # Regular binary: build both sides freely
    left,  rest, left_has_x  = _build_expr(rest)
    right, rest, right_has_x = _build_expr(rest)

    # If this subtree MUST contain x but neither child does, force left = x
    if must_contain_x and not left_has_x and not right_has_x:
        left        = x_sym
        left_has_x  = True

    expr_has_x = left_has_x or right_has_x

    try:
        result = BINARY_MAP[op](left, right)
    except Exception:
        result     = left
        expr_has_x = left_has_x

    return result, rest, expr_has_x


# ── Cheap structural metrics ──────────────────────────────────────────────────

def _depth(expr: sp.Expr) -> int:
    if expr.is_Atom:
        return 0
    return 1 + max(_depth(arg) for arg in expr.args)


def _node_count(expr: sp.Expr) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


# ── Guards ────────────────────────────────────────────────────────────────────

def _is_valid_expr(
    expr: sp.Expr,
    max_depth: int = 4,
    max_nodes: int = 12,
) -> bool:
    """
    Return True if the expression passes all POC quality filters.

    Checks:
      1. Must depend on x (not a constant).
      2. Structural limits: depth ≤ max_depth, nodes ≤ max_nodes.
         Note: sympy depth is affected by representation — e.g. x/(1-x)
         has depth 4 because -x is stored as Mul(-1, x) internally.
      3. No complex / undefined values (zoo, nan, oo, I).
      4. Must not be constant w.r.t. x (derivative ≠ 0).
      5. Every unary function node must have an argument that contains x.
         e.g. sin(3), exp(2), sqrt(5) are rejected.
      6. Power constraints on the bare variable x only:
           x**2  → OK     (in VALID_EXPONENTS)
           x**3  → OK     (in VALID_EXPONENTS)
           x**4  → REJECT (positive, not in VALID_EXPONENTS)
           x**0.5→ REJECT (non-integer)
           (1+x)**(-1) → OK  (base is compound, not bare x = division)
           sqrt(1+x)   → OK  (base is compound, not bare x)
    """
    if expr is None:
        return False
    if not expr.has(x_sym) or expr.is_number:
        return False
    if _depth(expr) > max_depth or _node_count(expr) > max_nodes:
        return False
    if expr.has(sp.zoo, sp.nan, sp.oo, sp.I):
        return False
    try:
        if sp.diff(expr, x_sym) == 0:
            return False
    except Exception:
        return False

    # ── Rule 5: every unary node must wrap a sub-expression containing x ─────
    for node in sp.preorder_traversal(expr):
        fname = type(node).__name__.lower()
        if fname in UNARY_MAP:
            arg = node.args[0]
            if not arg.has(x_sym):
                return False   # e.g. sin(3), exp(2) — reject

    # ── Rule 6: power constraints on the bare variable x ─────────────────────
    # Only restricts Pow nodes where the base is EXACTLY x_sym.
    #   x**2  → OK     (in VALID_EXPONENTS)
    #   x**3  → OK     (in VALID_EXPONENTS)
    #   x**4  → REJECT (positive, not in VALID_EXPONENTS)
    #   x**0.5→ REJECT (non-integer)
    #   (1+x)**(-1) → OK  (division; base is compound, not bare x)
    #   sqrt(1+x)   → OK  (base is compound, not bare x)
    for node in sp.preorder_traversal(expr):
        if node.is_Pow:
            base, exp_ = node.args
            if base == x_sym:                         # only restrict bare x
                if not exp_.is_Integer or int(exp_) not in VALID_EXPONENTS:
                    return False

    return True


def generate_random_function(cfg: Config, retries: int = 40) -> sp.Expr | None:
    """
    Sample a random function and return a valid sympy expr, or None.
    """
    for _ in range(retries):
        n_ops        = random.randint(*cfg.n_ops_range)
        nodes        = sample_tree(n_ops, cfg.max_depth, cfg.max_nodes)
        raw, _, has_x = _build_expr(nodes, must_contain_x=True)

        if not has_x or not raw.has(x_sym) or raw.is_number:
            continue
        if _depth(raw) > cfg.max_depth or _node_count(raw) > cfg.max_nodes:
            continue

        expr = run_with_timeout(
            sp.nsimplify, args=(raw,), kwargs={"rational": True},
            seconds=SIMPLIFY_TIMEOUT, fallback=None,
        )
        if expr is None:
            continue

        if not _is_valid_expr(expr, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes):
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
    • Rational p/q       →  ['/', *int_to_digit_tokens(p), *int_to_digit_tokens(q)]
    • Binary op          →  [op, left_tokens..., right_tokens...]
    • Unary fn           →  [fname, child_tokens...]
    • x, a               →  ['x'], ['a']
    """
    def _rec(e) -> list[str]:
        if e.is_Integer:
            return int_to_digit_tokens(int(e))

        if e.is_Rational:
            return ["/"] + int_to_digit_tokens(int(e.p)) + int_to_digit_tokens(int(e.q))

        if e.is_Number:
            return [str(e)]

        if e == x_sym: return ["x"]
        if e == a_sym: return ["a"]

        if e.is_Add:
            args = list(e.args)
            t = _rec(args[0])
            for arg in args[1:]:
                t = ["+"] + t + _rec(arg)
            return t

        if e.is_Mul:
            args = list(e.args)
            t = _rec(args[0])
            for arg in args[1:]:
                t = ["*"] + t + _rec(arg)
            return t

        if e.is_Pow:
            base, exp_ = e.args
            return ["**"] + _rec(base) + _rec(exp_)

        fname = type(e).__name__.lower()
        if fname in VOCAB_INDEX:
            return [fname] + _rec(e.args[0])

        return [str(e)]

    try:
        tokens = _rec(sp.nsimplify(expr, rational=True))
    except Exception:
        tokens = [str(expr)]

    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PREFIX → INFIX RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _consume_number(tokens: list[str], pos: int) -> tuple[sp.Expr, int]:
    sign = tokens[pos]; pos += 1
    digits: list[str] = []
    while pos < len(tokens) and tokens[pos].isdigit():
        digits.append(tokens[pos]); pos += 1
    value = int("".join(digits)) * (-1 if sign == "-" else 1)
    return sp.Integer(value), pos


def _prefix_tokens_to_expr(tokens: list[str], pos: int = 0) -> tuple[sp.Expr, int]:
    if pos >= len(tokens):
        return sp.Integer(1), pos

    tok = tokens[pos]

    if tok in ("+", "-") and pos + 1 < len(tokens) and tokens[pos + 1].isdigit():
        return _consume_number(tokens, pos)

    if tok == "x": return x_sym, pos + 1
    if tok == "a": return a_sym, pos + 1

    if tok in ("+", "-", "*", "/", "**"):
        left,  pos = _prefix_tokens_to_expr(tokens, pos + 1)
        right, pos = _prefix_tokens_to_expr(tokens, pos)
        return BINARY_MAP[tok](left, right), pos

    if tok in UNARY_MAP:
        child, pos = _prefix_tokens_to_expr(tokens, pos + 1)
        try:
            return UNARY_MAP[tok](child), pos
        except Exception:
            return child, pos

    return sp.Symbol(tok), pos + 1


def prefix_tokens_to_infix(tokens: list[str]) -> str:
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
    random.seed(seed)

    dataset:  list[dict] = []
    attempts: int        = 0
    t_wall               = time.perf_counter()

    _t: dict[str, list[float]] = {
        "generate_fn":   [],
        "taylor_coeffs": [],
        "prefix_tokens": [],
    }

    while len(dataset) < n_samples and attempts < n_samples * 200:
        try:
            attempts += 1

            # ── Stage: generate function ─────────────────────────────────────
            t0   = time.perf_counter()
            expr = generate_random_function(cfg)
            t_gen = time.perf_counter() - t0
            if expr is None:
                continue
            _t["generate_fn"].append(t_gen)

            # ── Stage: Taylor coefficients ───────────────────────────────────
            t0     = time.perf_counter()
            coeffs = compute_taylor_coefficients(expr, cfg.taylor_order)
            t_tay  = time.perf_counter() - t0
            if coeffs is None:
                continue
            if any(c.has(sp.zoo, sp.nan, sp.oo, sp.I) for c in coeffs):
                continue
            _t["taylor_coeffs"].append(t_tay)

            # ── Stage: prefix tokenisation ───────────────────────────────────
            t0        = time.perf_counter()
            fn_tokens = _expr_to_prefix_tokens(expr)
            c_tokens  = [_expr_to_prefix_tokens(c) for c in coeffs]
            t_tok     = time.perf_counter() - t0
            _t["prefix_tokens"].append(t_tok)

            # ── Assemble sample ──────────────────────────────────────────────
            depth = _depth(expr)
            nodes = _node_count(expr)
            n_ops = sum(1 for nd in sp.preorder_traversal(expr) if not nd.is_Atom)

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

        if VERBOSE:
            elapsed = time.perf_counter() - t_wall
            print(
                f"  [{count:05d}/{n_samples}]  "
                f"f(x)={str(expr):<40}  "
                f"depth={depth}  nodes={nodes:2d}  ops={n_ops}  "
                f"gen={t_gen*1e3:6.1f}ms  "
                f"taylor={t_tay*1e3:6.1f}ms  "
                f"tok={t_tok*1e3:5.2f}ms  "
                f"wall={elapsed:.1f}s"
            )

        if count % SAVE_EVERY == 0:
            _save_json(dataset, output_path)
            if not VERBOSE:
                print(f"  [checkpoint] {count}/{n_samples} samples  →  {output_path}")

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
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  TEST: prefix ↔ infix roundtrip")
    print(SEP)

    random.seed(seed)
    passed = failed = tested = 0

    while tested < n:
        expr = generate_random_function(cfg)
        if expr is None:
            continue

        tokens        = _expr_to_prefix_tokens(expr)
        reconstructed = prefix_tokens_to_infix(tokens)
        try:
            recon_expr = sp.sympify(reconstructed, locals={"x": x_sym, "a": a_sym})
            ok_func    = sp.simplify(expr - recon_expr) == 0
        except Exception:
            ok_func    = False

        coeffs    = compute_taylor_coefficients(expr, order=4)
        ok_coeffs = True
        if coeffs:
            for n_idx, cn in enumerate(coeffs):
                if cn.has(sp.zoo, sp.nan, sp.oo, sp.I):
                    continue
                c_toks  = _expr_to_prefix_tokens(cn)
                c_recon = prefix_tokens_to_infix(c_toks)
                try:
                    c_expr = sp.sympify(c_recon, locals={"x": x_sym, "a": a_sym})
                    c_ok   = sp.simplify(cn - c_expr) == 0
                except Exception:
                    c_ok = False
                if not c_ok:
                    ok_coeffs = False
                    print(f"    [COEFF FAIL] coeff{n_idx}: {cn}  →  {c_recon}")
                    break

        tested += 1
        if ok_func and ok_coeffs:
            passed += 1
            print(f"  [PASS #{tested:02d}]  f(x) = {expr}")
            print(f"            prefix = {tokens}")
        else:
            diff = sp.simplify(expr - recon_expr) if not ok_func else "coeff mismatch"
            print(f"  [FAIL #{tested:02d}]  f(x)={expr}  diff={diff}")

    print("-" * 70)
    print(f"  Result: {passed}/{tested} passed,  {failed} failed")
    print("  ALL ROUNDTRIP TESTS PASSED ✓" if failed == 0 else "  SOME TESTS FAILED ✗")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(cfg: Config, n_trials: int = 50, seed: int = 123) -> None:
    random.seed(seed)
    stages = [
        "1_sample_tree",        "2_build_expr",
        "3_nsimplify",          "4_taylor_coefficients",
        "5_expr_prefix_tokens", "6_coeff_prefix_tokens",
        "7_prefix_to_infix",    "8_full_build_sample",
    ]
    timings: dict[str, list[float]] = {s: [] for s in stages}
    completed = attempts = 0

    while completed < n_trials and attempts < n_trials * 200:
        try:
            attempts += 1
            n_ops = random.randint(*cfg.n_ops_range)

            t0 = time.perf_counter()
            nodes = sample_tree(n_ops, cfg.max_depth, cfg.max_nodes)
            timings["1_sample_tree"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            raw_expr, _, _ = _build_expr(nodes, must_contain_x=True)
            timings["2_build_expr"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            expr = run_with_timeout(
                sp.nsimplify, args=(raw_expr,), kwargs={"rational": True},
                seconds=SIMPLIFY_TIMEOUT, fallback=None,
            )
            timings["3_nsimplify"].append(time.perf_counter() - t0)
            if expr is None or not _is_valid_expr(expr, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes):
                continue

            t0     = time.perf_counter()
            coeffs = compute_taylor_coefficients(expr, cfg.taylor_order)
            timings["4_taylor_coefficients"].append(time.perf_counter() - t0)
            if coeffs is None or any(c.has(sp.zoo, sp.nan, sp.oo, sp.I) for c in coeffs):
                continue

            t0          = time.perf_counter()
            expr_tokens = _expr_to_prefix_tokens(expr)
            timings["5_expr_prefix_tokens"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for cn in coeffs:
                _expr_to_prefix_tokens(cn)
            timings["6_coeff_prefix_tokens"].append(
                (time.perf_counter() - t0) / len(coeffs)
            )

            t0 = time.perf_counter()
            prefix_tokens_to_infix(expr_tokens)
            timings["7_prefix_to_infix"].append(time.perf_counter() - t0)

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
            "n_ops":        n_ops,
            "tree_depth":   _depth(expr),
            "tree_nodes":   _node_count(expr),
            "taylor_order": cfg.taylor_order,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════╗
# ║                  U S E R   C O N F I G              ║
# ╚══════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    VERBOSE           = False
    SAVE_EVERY        = 100       # save JSON checkpoint every 100 samples
    SIMPLIFY_TIMEOUT  = 5

    N_SAMPLES   = 10000
    SEED        = random.randint(0, 100000)
    OUTPUT_PATH = "datasets/taylor_dataset.json"

    cfg = Config(
        n_ops_range  = (2, 5),
        max_depth    = 4,
        max_nodes    = 12,
        taylor_order = 4,
        x_bias       = 4.0,
    )

    print("=" * 68)
    print("  Taylor Series Dataset Generator")
    print("=" * 68)
    print(f"  samples    : {N_SAMPLES}")
    print(f"  seed       : {SEED}")
    print(f"  output     : {OUTPUT_PATH}")
    print(f"  save_every : {SAVE_EVERY}")
    print(f"  timeout    : {SIMPLIFY_TIMEOUT}s per sympy call")
    print("=" * 68 + "\n")

    t0      = time.perf_counter()
    dataset = generate_dataset(
        cfg         = cfg,
        n_samples   = N_SAMPLES,
        seed        = SEED,
        output_path = OUTPUT_PATH,
    )

    print(f"\n  Final save → {OUTPUT_PATH}")
    print(f"  Total samples : {len(dataset)}")
    print(f"  Wall time     : {time.perf_counter() - t0:.2f}s")