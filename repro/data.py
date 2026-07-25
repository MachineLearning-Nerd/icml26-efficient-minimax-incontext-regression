"""Data-generating functions for the in-context nonparametric regression model.

We need ground-truth regression functions with *controlled* Hölder regularity so
that the local-polynomial rate n^{-2 alpha/(2 alpha+d)} (Theorem 2.5) can be
tested. The canonical exactly-alpha-Hölder "cusp" function is

    m(x) = M * ||x - x0||^alpha   (alpha > 0, non-integer alpha preferred)

which is alpha-Hölder on [0,1]^d: |m(x)-m(x')| <= C ||x-x'||^alpha, with the
floor(alpha)-th derivative being (alpha-floor(alpha))-Hölder. For non-integer
alpha this genuinely saturates the Hölder smoothness (it is NOT smoother), so it
is the worst-case-appropriate function for the rate test.

A random-Fourier-series family (matching the authors' simulation model) is also
provided as a secondary, smoother-than-alpha check.
"""
from __future__ import annotations

import numpy as np


def holder_cusp(x: np.ndarray, alpha: float, center: np.ndarray | None = None,
                M: float = 1.0) -> np.ndarray:
    """m(x) = M * ||x - center||^alpha, an exactly alpha-Hölder function."""
    d = x.shape[-1]
    if center is None:
        center = np.full(d, 0.5)
    diff = x - center
    r = np.sqrt((diff ** 2).sum(-1))
    return M * r ** alpha


def holder_constant_estimate(alpha: float, d: int, M_scale: float = 1.0,
                             n_pairs: int = 4000, seed: int = 1) -> float:
    """Estimate the Hölder constant: smallest C with |m(x)-m(x')| <= C||x-x'||^alpha.
    Returns the supremum of the ratio over random pairs (a finite sample upper bound)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n_pairs, d))
    xp = rng.uniform(0, 1, size=(n_pairs, d))
    m = holder_cusp(x, alpha, M=M_scale)
    mp = holder_cusp(xp, alpha, M=M_scale)
    dist = np.sqrt(((x - xp) ** 2).sum(-1))
    mask = dist > 1e-9
    ratios = np.abs(m[mask] - mp[mask]) / (dist[mask] ** alpha)
    return float(ratios.max())


def sample_regression_data(rng: np.random.Generator, n: int, d: int,
                           m_fn, sigma: float, n_queries: int = 1):
    """X ~ U[0,1]^d, Y = m(X) + eps. Returns context (X,Y) and a query point with
    its noiseless m value (so excess risk = E[(m(Xq) - f)^2] can be measured)."""
    X = rng.uniform(0, 1, size=(n, d))
    Y = m_fn(X) + sigma * rng.standard_normal(n)
    Xq = rng.uniform(0, 1, size=(n_queries, d))
    mq = m_fn(Xq)
    return X, Y, Xq, mq


def random_fourier_series(rng: np.random.Generator, x: np.ndarray, alpha: float,
                          d: int, n_features: int = 256, freq_scale: float = 3.14,
                          beta: float | None = None) -> np.ndarray:
    """Random Fourier series with spectral decay (1+||w||^2)^{-beta/2}.

    The authors use beta = alpha + 1.5 in d=3; a Sobolev-beta function embeds into
    Hölder alpha when beta > alpha + d/2. We default to beta = alpha + d/2 + 0.5
    so the realised regularity is >= alpha (locpol tuned for alpha then achieves
    a rate at least as fast as n^{-2a/(2a+d)})."""
    if beta is None:
        beta = alpha + d / 2 + 0.5
    m = x.shape[0]
    omega = freq_scale * rng.standard_normal((n_features, d))
    phi = 2 * np.pi * rng.uniform(0, 1, n_features)
    amp = (1 + (omega ** 2).sum(1)) ** (-beta / 2)
    a = amp * rng.standard_normal(n_features)
    proj = x @ omega.T + phi  # (m, n_features)
    return (a * np.cos(proj)).sum(1) / np.sqrt(n_features)
