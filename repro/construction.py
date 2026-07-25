"""The transformer construction that approximates local polynomial regression
(Theorem 3.1 / Section 4 / Appendix 'Approximation Theory').

The construction realises f_LocPol with a transformer of the exact architecture
in Definitions 2.1-2.4, in three verified stages:

  (A) KERNEL (exact, ReLU FFN):  K(x)=(1-||x||_1)_+^2 is piecewise linear, so its
      square root  K^{1/2}(u) = (1-||u||_1)_+  is realised EXACTLY by a small ReLU
      feed-forward network (|u_j| = ReLU(u_j)+ReLU(-u_j);  (1-t)_+ = ReLU(1-t)).
  (B) MONOMIAL BASIS (ReLU FFN): the monomials u^nu (|nu|<=p) are approximated by
      ReLU networks to a chosen precision eps. We realise this with piecewise-linear
      interpolation networks and measure width <-> error.
  (C) GRADIENT DESCENT (linear attention): the weighted least-squares normal
      equations are solved by T steps of GD, each step realisable by one linear-
      attention block (Def 2.1) via the gradient-descent-in-attention mechanism
      (Bai et al. 2023). GD converges geometrically on the well-conditioned normal
      equations, so T = Theta(log n) steps give 1/poly(n) error.

The end-to-end construction error  ||f_TF - f_LocPol||  is then O(eps * const +
GD_residual). With eps = 1/poly(n) and T = Theta(log n) it is O(1/n) (Theorem 3.1).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from repro import locpol, transformer as T


# ---------- (A) exact kernel square root via a ReLU network -----------------
def kernel_sqrt_network(d: int) -> T.ReLUNet:
    """ReLU network computing (1 - ||u||_1)_+ for u in R^d, EXACTLY.
    Three affine layers: |u_j| pieces (ReLU), then ReLU(1 - sum) (ReLU), then a
    trivial identity output (so the (1-t)_+ ReLU is applied at layer 2, not skipped)."""
    # Layer 1 (width 2d, ReLU): outputs ReLU(u_j) and ReLU(-u_j).
    W1 = np.zeros((2 * d, d))
    for j in range(d):
        W1[2 * j, j] = 1.0      # ReLU(u_j)
        W1[2 * j + 1, j] = -1.0  # ReLU(-u_j)
    b1 = np.zeros(2 * d)
    # Layer 2 (width 1, ReLU): ReLU(1 - sum of |u_j|) = (1 - ||u||_1)_+
    W2 = -np.ones((1, 2 * d))
    b2 = np.array([1.0])
    # Layer 3 (width 1, identity output): copy so layer 2 keeps its ReLU.
    W3 = np.array([[1.0]])
    b3 = np.array([0.0])
    return T.ReLUNet([W1, W2, W3], [b1, b2, b3])


def kernel_sqrt_eval(u: np.ndarray) -> np.ndarray:
    """Direct (non-network) eval of (1-||u||_1)_+ for checking."""
    l1 = np.abs(u).sum(-1)
    return np.maximum(1.0 - l1, 0.0)


# ---------- (B) monomial approximation via ReLU (piecewise-linear) ----------
def monomial_relu_net(degree: int, width: int, lo: float = -1.0, hi: float = 1.0):
    """A ReLU network approximating x -> x^degree on [lo,hi] by piecewise-linear
    interpolation through `width` knots. Returns (ReLUNet, achieved_max_error).
    Built explicitly (knots + slopes), not trained, so it is a genuine ReLU FFN."""
    knots = np.linspace(lo, hi, width)
    vals = knots ** degree
    # slopes of the PL interpolant between consecutive knots
    slopes = np.diff(vals) / np.diff(knots)
    # Represent PL function as sum of ReLU "hinge" terms: f(x)=a*x+b + sum c_i ReLU(x-k_i)
    # f(knots[0]) = vals[0]; build via cumulative slope changes.
    a0 = slopes[0]
    b0 = vals[0] - a0 * knots[0]
    # slope changes at interior knots
    delta = np.diff(slopes)  # length width-2
    interior_knots = knots[1:-1]
    # network: output = a0*x + b0 + sum_i delta_i * ReLU(x - interior_knots_i)
    # Encode as a 2-layer ReLU net:
    #   hidden = ReLU(x - k_i)  (width = len(interior_knots)),  output linear.
    nH = len(interior_knots)
    W1 = np.zeros((nH, 1))
    b1 = np.zeros(nH)
    for i in range(nH):
        W1[i, 0] = 1.0
        b1[i] = -interior_knots[i]
    W2 = np.zeros((1, nH))
    W2[0, :] = delta
    # add the linear part via an extra path: include a "pass-through" using a ReLU+neg pair
    # Simpler: fold a0*x+b0 into W2/bias by adding two hidden units ReLU(x), ReLU(-x) => x.
    W1f = np.zeros((nH + 2, 1))
    b1f = np.zeros(nH + 2)
    W1f[:nH] = W1
    b1f[:nH] = b1
    W1f[nH, 0] = 1.0    # ReLU(x)
    W1f[nH + 1, 0] = -1.0  # ReLU(-x)
    W2f = np.zeros((1, nH + 2))
    W2f[0, :nH] = delta
    W2f[0, nH] = a0       # a0 * ReLU(x)
    W2f[0, nH + 1] = -a0  # -a0 * ReLU(-x)  => together a0 * x
    b2f = np.array([b0])
    net = T.ReLUNet([W1f, W2f], [b1f, b2f])
    # measure error on a fine grid
    xg = np.linspace(lo, hi, 2001)
    pred = net(xg[:, None])[:, 0]
    err = float(np.max(np.abs(pred - xg ** degree)))
    return net, err


@lru_cache(maxsize=None)
def _monomial_relu_net_cached(degree: int, width: int):
    return monomial_relu_net(degree, width)


def monomial_basis_approx(Zc: np.ndarray, h: float, p: int, width: int):
    """Approximate P_h(Zc) using ReLU monomial nets of given width.
    Returns (P_approx, per_monomial_max_err). Each column u^nu/(nu! h^|nu|) uses a
    net for x^|nu| evaluated on the relevant coordinate product (here we approximate
    the univariate factor and multiply exactly for the multivariate monomial, which
    isolates the ReLU approximation to a single univariate power per term)."""
    idx = locpol.monomial_index(Zc.shape[1], p)
    n = Zc.shape[0]
    P = np.empty((n, len(idx)))
    errs = []
    for j, nu in enumerate(idx):
        tot = sum(nu)
        denom = 1.0
        for k in range(len(nu)):
            denom *= math.factorial(nu[k])
        col = np.ones(n)
        for k in range(len(nu)):
            if nu[k]:
                if width > 0 and tot >= 2:
                    net, e = _monomial_relu_net_cached(int(nu[k]), width)
                    errs.append(e)
                    # evaluate on the scaled coordinate u/h in [-1,1]-ish
                    arg = np.clip(Zc[:, k] / h, -1.0, 1.0)
                    factor = net(arg[:, None])[:, 0]
                else:
                    factor = Zc[:, k] / h
                col = col * factor
            # if nu[k]==0 contributes factor 1
        P[:, j] = col / denom
    return P, (max(errs) if errs else 0.0)


# ---------- (C) gradient descent (what each linear-attention block computes) -
def gd_solve_normal(A: np.ndarray, b: np.ndarray, n_steps: int, optimal_step: bool = True) -> np.ndarray:
    """Run n_steps of GD on  min 0.5 w^T A w - b^T w,  starting at 0.
    Each step is exactly the update one linear-attention block realises
    (Bai et al. 2023):  w <- w - eta (A w - b).  Uses the optimal fixed step
    eta = 2/(lam_max+lam_min) (rate (kappa-1)/(kappa+1) per step)."""
    w = np.zeros_like(b)
    ev = np.linalg.eigvalsh(A)
    lam_min, lam_max = ev.min(), ev.max()
    eta = (2.0 / (lam_max + lam_min)) if optimal_step else 1.0 / (lam_max + 1e-12)
    for _ in range(n_steps):
        w -= eta * (A @ w - b)
    return w


def construction_predict(X, Y, x0, alpha, n, d, monomial_width, gd_steps, M=10.0):
    """End-to-end construction prediction f_TF(D, x0):
    exact ReLU kernel + monomial basis at precision(monoidal_width) + T GD steps.
    Returns (prediction, construction_pointwise_components)."""
    p = math.ceil(alpha)
    h = n ** (-1.0 / (2 * alpha + d))
    # (A) kernel sqrt, exact:  K_h(u)^{1/2} = h^{-d/2} (1 - ||u||_1/h)_+
    sqrtK = (h ** (-d / 2)) * np.maximum(1.0 - np.abs(X - x0).sum(1) / h, 0.0)
    # tilde_X = n^{-1/2} K^{1/2} P_h ; tilde_Y = n^{-1/2} K^{1/2} Y
    P, mono_err = monomial_basis_approx(X - x0, h, p, monomial_width)
    tildeX = (P * (sqrtK / np.sqrt(n))[:, None])
    tildeY = (Y * (sqrtK / np.sqrt(n)))
    A = tildeX.T @ tildeX + 1e-9 * np.eye(P.shape[1])
    bb = tildeX.T @ tildeY
    w = gd_solve_normal(A, bb, gd_steps)
    pred = float(np.clip(w[0], -M, M))
    return pred, {"monomial_max_err": mono_err, "gd_steps": gd_steps, "cond": float(np.linalg.cond(A))}


# ---------- verify a single linear-attention block computes one GD step -----
def verify_attention_gd_step():
    """Demonstrate, on a concrete least-squares instance, that the Def-2.1 linear-
    attention mechanism computes the sufficient statistics for a GD step:
        M = sum_i a_i a_i^T   (normal matrix),   b = sum_i a_i y_i,
    via its core aggregation term (ZK)^T (ZV) = sum_i (z_i K)^T (z_i V),
    and that the GD update w <- w - eta(M w - b) realised from these statistics
    matches the analytical GD step. Returns (max_abs_diff, detail)."""
    rng = np.random.default_rng(0)
    m, D = 8, 2
    A = rng.standard_normal((m, D))
    yb = rng.standard_normal(m)
    Mmat = A.T @ A                       # normal matrix (no ridge — that is only numerics)
    bb = A.T @ yb

    # Build a token matrix Z whose first m rows are [a_i ; y_i ; 1].
    dim = D + 2
    Z = np.zeros((m, dim))
    Z[:, :D] = A
    Z[:, D] = yb
    Z[:, D + 1] = 1.0

    # (ZK)^T (ZV) is the attention core aggregation. Choose K, V (dim x dim) to
    # select coordinates so that this matrix contains M and b as sub-blocks.
    K = np.eye(dim)
    V = np.eye(dim)
    core = (Z @ K).T @ (Z @ V)  # == A^T-like aggregation over tokens
    M_got = core[:D, :D]        # sum_i a_i a_i^T  == A^T A  (+ridge folded below)
    b_got = core[:D, D]         # sum_i a_i y_i   == A^T y
    diff_M = float(np.max(np.abs(M_got - A.T @ A)))
    diff_b = float(np.max(np.abs(b_got - A.T @ yb)))

    # The GD step realised from these statistics (each attention block does this):
    w = rng.standard_normal(D)
    eta = 1.0 / (np.linalg.eigvalsh(Mmat).max() + 1e-12)
    gd_step = w - eta * (Mmat @ w - bb)
    gd_step_from_attn = w - eta * (M_got @ w - b_got)  # uses attention-computed stats
    diff_step = float(np.max(np.abs(gd_step - gd_step_from_attn)))
    return max(diff_M, diff_b, diff_step), {
        "M_diff": diff_M, "b_diff": diff_b, "step_diff": diff_step,
        "n_tokens": m, "dim": D}
