"""Exact architecture constants of the paper's transformer construction
(Theorem 3.2 / Definitions 2.1-2.4).

All sizes are taken verbatim from the LaTeX source and reduced to pure-Python
functions of the smoothness ``alpha`` and covariate dimension ``d``. They depend
on ``n`` ONLY through the number of blocks ``L = ceil(C log(en))`` and the
per-entry magnitude bound ``B = C n^2``; the per-block parameter *count* is
independent of ``n``.
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement


def poly_degree(alpha: float) -> int:
    """p := ceil(alpha)  (Theorem 2.5 / 3.2)."""
    return math.ceil(alpha)


def monomial_dim(d: int, p: int) -> int:
    """D := binom(d+p, p) — number of monomials of total degree <= p in d dims."""
    return math.comb(d + p, p)


def embed_dim(d: int, alpha: float) -> int:
    """delta := 2d + 2D + 5  (Theorem 3.2)."""
    p = poly_degree(alpha)
    D = monomial_dim(d, p)
    return 2 * d + 2 * D + 5


def ffn_width(d: int, alpha: float) -> int:
    """d_ffn := 6(D+1)(14+p)  (Theorem 3.2)."""
    p = poly_degree(alpha)
    D = monomial_dim(d, p)
    return 6 * (D + 1) * (14 + p)


def n_blocks(n: int, C: float = 1.0) -> int:
    """L := ceil(C log(en))  (Theorem 3.2). C is an unspecified absolute constant;
    we expose it so the count's *scaling* with n can be checked independent of C."""
    return math.ceil(C * math.log(math.e * n))


def per_block_param_count(d: int, alpha: float) -> int:
    """Scalar parameter COUNT of one transformer block (Defs 2.1 + 2.2).

    Linear attention: Q,K,V in R^{delta x delta}                 -> 3 delta^2
    FFN:            W1 (d_ffn x delta), W2 (delta x d_ffn),
                    b1 (d_ffn), b2 (delta)                        -> 2 delta d_ffn + d_ffn + delta
    """
    delta = embed_dim(d, alpha)
    dffn = ffn_width(d, alpha)
    attn = 3 * delta * delta
    ffn = 2 * delta * dffn + dffn + delta
    return attn + ffn


def total_param_count(n: int, d: int, alpha: float, C: float = 1.0) -> int:
    """Total scalar parameter COUNT = L * per-block  (independent of the magnitude B)."""
    return n_blocks(n, C) * per_block_param_count(d, alpha)


def param_bound(n: int, C: float = 1.0) -> float:
    """B := C n^2 — per-entry MAGNITUDE bound (Definition 2.4: 'every entry of
    each parameter in theta bounded in absolute value by B'). NOT a count."""
    return C * n * n


# --- comparison baselines (stated in paper, Section 'Main Results') --------
def shen_params(n: int) -> float:
    """Shen et al. (2025): Theta(n) parameters."""
    return float(n)


def kim_params(n: int, d: int, alpha: float) -> float:
    """Kim et al. (2024): Theta(n^{d/(2 alpha + d)}) parameters."""
    return n ** (d / (2 * alpha + d))


def our_pretraining(n: int, d: int, alpha: float, C: float = 1.0) -> float:
    """Gamma >= C n^{2 alpha/(2 alpha+d)} log^3(en)  (Theorem 3.2)."""
    return C * n ** (2 * alpha / (2 * alpha + d)) * math.log(math.e * n) ** 3


def shen_pretraining(n: int, d: int, alpha: float) -> float:
    """Shen et al. (2025): Omega(n^{(6 alpha+d)/(2 alpha+d)} log n)."""
    return n ** ((6 * alpha + d) / (2 * alpha + d)) * math.log(math.e * n)


def kim_pretraining(n: int, d: int, alpha: float) -> float:
    """Kim et al. (2024): Omega(n^{(2 alpha+2d)/(2 alpha+d)} log n)."""
    return n ** ((2 * alpha + 2 * d) / (2 * alpha + d)) * math.log(math.e * n)


def minimax_rate(n: int, d: int, alpha: float) -> float:
    """n^{-2 alpha/(2 alpha+d)}  (the minimax MSE rate)."""
    return n ** (-2 * alpha / (2 * alpha + d))
