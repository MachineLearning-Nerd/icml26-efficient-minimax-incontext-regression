"""Faithful local polynomial estimator (Theorem 2.5 construction).

Implements the truncated local polynomial estimator from Section 'Local Polynomial
Estimators':
  - degree p = ceil(alpha)
  - kernel K(u) = (1 - ||u||_1)_+^2   (the paper's specific kernel)
  - bandwidth h = n^{-1/(2 alpha+d)}
  - monomial basis P_h(x-x0)_nu = (x-x0)^nu / (nu! h^{|nu|}), |nu| <= p
  - weighted least squares:  w* = argmin (1/n) sum K_h(Xi-x0)(Yi - w^T P_h(Xi-x0))^2
  - estimator = truncated first component  (-M) v w*_{1} ^ M

Also exposes a pure-gradient-descent solver of the same normal equations, used by
the construction claim (C2/C6) to show attention-implemented GD matches the
closed form.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def monomial_index(d: int, p: int) -> list[tuple[int, ...]]:
    """Multi-indices nu in N_0^d with |nu| <= p, in increasing lexicographic order
    (so the first is the all-zero index => the constant term => first component)."""
    idx = []
    for total in range(0, p + 1):
        for comb in itertools.combinations_with_replacement(range(d), total):
            nu = [0] * d
            for j in comb:
                nu[j] += 1
            idx.append(tuple(nu))
    idx.sort(key=lambda nu: (sum(nu), nu))
    return idx


def design_matrix(Xc: np.ndarray, h: float, p: int) -> np.ndarray:
    """P_h(Xc): rows are (Xc_i^nu / (nu! h^{|nu|})) for each multi-index nu, |nu|<=p.
    Xc has shape (n, d) = centred covariates (X_i - x0)."""
    n, d = Xc.shape
    idx = monomial_index(d, p)
    P = np.empty((n, len(idx)))
    for j, nu in enumerate(idx):
        denom = 1.0
        for k in range(d):
            denom *= math.factorial(nu[k])
        col = np.ones(n)
        for k in range(d):
            if nu[k]:
                col = col * (Xc[:, k] ** nu[k])
        P[:, j] = col / (denom * h ** sum(nu))
    return P


def kernel_weights(Xc: np.ndarray, h: float) -> np.ndarray:
    """K_h(Xi-x0) = (1 - ||Xi-x0||_1 / h)_+^2 / h^d.  (K(u)=(1-||u||_1)_+^2.)"""
    n, d = Xc.shape
    l1 = np.abs(Xc).sum(1)
    k = np.maximum(1.0 - l1 / h, 0.0) ** 2
    return k / (h ** d)


def locpol_weights_solve(X: np.ndarray, Y: np.ndarray, x0: np.ndarray,
                         alpha: float, n: int, d: int, p: int | None = None,
                         ridge: float = 1e-9) -> np.ndarray:
    """Closed-form w* = (P^T W P + ridge I)^{-1} P^T W Y. Returns the full w* vector."""
    if p is None:
        p = math.ceil(alpha)
    h = n ** (-1.0 / (2 * alpha + d))
    Xc = X - x0
    P = design_matrix(Xc, h, p)
    W = kernel_weights(Xc, h)
    A = (P * W[:, None]).T @ P + ridge * np.eye(P.shape[1])
    b = (P * W[:, None]).T @ Y
    return np.linalg.solve(A, b)


def locpol_predict(X: np.ndarray, Y: np.ndarray, x0: np.ndarray, alpha: float,
                   n: int, d: int, p: int | None = None, M: float = 10.0,
                   ridge: float = 1e-9) -> float:
    """Truncated local polynomial prediction at x0: (-M) v w*_{1} ^ M."""
    w = locpol_weights_solve(X, Y, x0, alpha, n, d, p, ridge)
    val = w[0]
    return float(np.clip(val, -M, M))


def locpol_predict_gd(X: np.ndarray, Y: np.ndarray, x0: np.ndarray, alpha: float,
                      n: int, d: int, p: int | None = None, n_steps: int = 0,
                      M: float = 10.0, ridge: float = 1e-9) -> tuple[float, np.ndarray]:
    """Gradient-descent solution of the SAME normal equations (used to show the
    attention-implemented GD path matches the closed form). Returns (prediction, w).
    n_steps<=0 picks a default of ceil(C log(en)) steps (matching the construction)."""
    if p is None:
        p = math.ceil(alpha)
    if n_steps <= 0:
        n_steps = math.ceil(3 * math.log(math.e * n))
    h = n ** (-1.0 / (2 * alpha + d))
    Xc = X - x0
    P = design_matrix(Xc, h, p)
    W = kernel_weights(Xc, h)
    A = (P * W[:, None]).T @ P + ridge * np.eye(P.shape[1])
    b = (P * W[:, None]).T @ Y
    w = np.zeros_like(b)
    lam = np.linalg.eigvalsh(A).max() + 1e-12
    eta = 1.0 / lam
    for _ in range(n_steps):
        w -= eta * (A @ w - b)
    return float(np.clip(w[0], -M, M)), w


def excess_error(pred: float, m_true: float) -> float:
    """E[(m(Xq) - f)^2] contribution (noiseless query)."""
    return float((pred - m_true) ** 2)
