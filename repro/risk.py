"""Population-risk helpers for in-context regression.

R(f) - sigma^2 = E[(m(X_{n+1}) - f(D_n, X_{n+1}))^2]  (excess prediction risk).
Measured by Monte Carlo: draw many (context, query) sets from a fixed alpha-Hölder
ground truth and average the squared prediction error at the (noiseless) query.
"""
from __future__ import annotations

import numpy as np

from repro import data


def excess_risk(predictor, m_fn, n, d, sigma, reps, seed, **pred_kwargs):
    """Monte-Carlo estimate of R(f)-sigma^2 for a predictor(D, Xq)->values."""
    rng = np.random.default_rng(seed)
    errs = np.empty(reps)
    for t in range(reps):
        X, Y, Xq, mq = data.sample_regression_data(rng, n, d, m_fn, sigma, 1)
        pred = predictor(X, Y, Xq[0], **pred_kwargs)
        errs[t] = (pred - mq[0]) ** 2
    return float(errs.mean()), float(errs.std() / np.sqrt(reps))
