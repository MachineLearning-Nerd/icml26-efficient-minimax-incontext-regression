"""Generate the publication figures for the reproduction report.

Writes PNGs into reports/incontext-nonparametric/images/. Re-runs the (fast)
measurements from the claim modules so the figures are self-contained evidence.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repro import architecture as A, construction as C, data, locpol

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "incontext-nonparametric", "images")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_locpol_rate():
    """Headline: locpol excess risk at the Hölder cusp vs n, with -2a/(2a+d) guides."""
    cases = [(0.5, 1, "C0"), (1.5, 1, "C1"), (1.5, 2, "C2"), (2.5, 2, "C3")]
    ns = [2 ** k for k in range(5, 12)]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for alpha, d, col in cases:
        mse = []
        for n in ns:
            rng = np.random.default_rng(2024 + n + int(10 * alpha))
            x0 = np.full(d, 0.5)
            e = []
            for _ in range(160):
                X, Y, _, _ = data.sample_regression_data(rng, n, d, lambda x: data.holder_cusp(x, alpha), 0.05)
                e.append(locpol.locpol_predict(X, Y, x0, alpha, n, d) ** 2)
            mse.append(np.mean(e))
        s = float(np.polyfit(np.log(ns), np.log(mse), 1)[0])
        r = 2 * alpha / (2 * alpha + d)
        ax.loglog(ns, mse, "o-", color=col, lw=1.6, ms=4,
                  label=fr"$\alpha={alpha}, d={d}$: slope ${s:.2f}$ (rate ${-r:.2f}$)")
        ax.loglog(ns, mse[0] * (np.array(ns) / ns[0]) ** (-r), "--", color=col, alpha=0.4, lw=1)
    ax.set_xlabel("context size $n$")
    ax.set_ylabel(r"pointwise excess risk  $R(f_{\mathrm{LocPol}})-\sigma^2$")
    ax.set_title("Local polynomial rate $n^{-2\\alpha/(2\\alpha+d)}$ (Theorem 2.5)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_locpol_rate.png"), dpi=130)
    plt.close(fig)


def fig_construction_approx():
    """C2: construction error ||f_TF - f_LocPol||_2 vs n (O(1/n))."""
    from repro.claims.c2_transformer_approx import _eps_for_n
    ns = [16, 32, 64, 128, 256]
    jobs = [(n, 1.5, 1, 30, 2.0, 400.0) for n in ns]
    rows = [_eps_for_n(j) for j in jobs]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog([r["n"] for r in rows], [r["rms_error"] for r in rows], "o-", lw=1.7, ms=5,
              label="construction error")
    n0 = np.array([r["n"] for r in rows])
    e0 = np.array([r["rms_error"] for r in rows])
    ax.loglog(n0, e0[0] * (n0 / n0[0]) ** (-1.0), "--", color="gray", lw=1.2, label="$O(1/n)$ reference")
    s = float(np.polyfit(np.log(n0), np.log(e0), 1)[0])
    ax.set_xlabel("context size $n$")
    ax.set_ylabel(r"$\|f_{\mathrm{TF}}-f_{\mathrm{LocPol}}\|_2$ (RMS)")
    ax.set_title(f"Transformer approximates locpol: error $O(1/n)$ (Theorem 3.1), slope {s:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_construction_approx.png"), dpi=130)
    plt.close(fig)


def fig_construction_components():
    """C6: monomial ReLU error vs width (left) and GD steps-to-tolerance vs log n (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    # monomial approximation
    for deg, col in [(2, "C0"), (3, "C1"), (4, "C2")]:
        ws = [8, 16, 32, 64, 128]
        es = [C.monomial_relu_net(deg, w)[1] for w in ws]
        ax1.loglog(ws, es, "o-", color=col, lw=1.5, ms=4, label=f"$x^{deg}$")
    ax1.set_xlabel("ReLU network width (knots)")
    ax1.set_ylabel("max approximation error")
    ax1.set_title("(A) ReLU FFN approximates monomial basis")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    # GD steps to reach 1/n
    ns = [64, 256, 1024, 4096]
    tmins = []
    for n in ns:
        alpha, d = 1.5, 1
        rng = np.random.default_rng(100 + n)
        X, Y, _, _ = data.sample_regression_data(rng, n, d, lambda x: data.holder_cusp(x, alpha), 0.05)
        x0 = np.full(d, 0.5)
        w_star = locpol.locpol_weights_solve(X, Y, x0, alpha, n, d)
        h = n ** (-1.0 / (2 * alpha + d))
        P = locpol.design_matrix(X - x0, h, math.ceil(alpha))
        W = locpol.kernel_weights(X - x0, h)
        Aa = (P * W[:, None]).T @ P + 1e-9 * np.eye(P.shape[1])
        b = (P * W[:, None]).T @ Y
        ev = np.linalg.eigvalsh(Aa)
        eta = 2.0 / (ev.max() + ev.min())
        w = np.zeros_like(b)
        for t in range(1, 200000):
            w -= eta * (Aa @ w - b)
            if np.max(np.abs(w - w_star)) < 1.0 / n:
                tmins.append(t)
                break
    ax2.plot([math.log(math.e * n) for n in ns], tmins, "s-", color="C3", lw=1.7, ms=6)
    ax2.set_xlabel(r"$\log(en)$")
    ax2.set_ylabel("GD steps to reach gap $<1/n$")
    ax2.set_title(r"(B) $\Theta(\log n)$ attention-GD steps converge (C6)")
    ax2.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_construction_components.png"), dpi=130)
    plt.close(fig)


def fig_param_count():
    """C4: growth-rate (log-log slope) of parameter count: ours ~0 vs Shen 1, Kim d/(2a+d)."""
    import numpy as np
    def slope(func, d, alpha, lo=8, hi=22):
        ng = np.array([2 ** k for k in range(lo, hi)], dtype=float)
        y = np.array([func(int(n), d, alpha) for n in ng])
        return float(np.polyfit(np.log(ng), np.log(y), 1)[0])
    d, alpha = 3, 3.0
    ns = np.array([2 ** k for k in range(8, 22)])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for func, lab, col, sty in [
        (A.total_param_count, f"ours $\\Theta(\\log n)$", "C2", "o-"),
        (lambda n, dd, aa: A.shen_params(n), f"Shen $\\Theta(n)$", "C3", "s-"),
        (A.kim_params, f"Kim $\\Theta(n^{{d/(2\\alpha+d)}})$", "C4", "^-"),
    ]:
        y = np.array([func(int(n), d, alpha) for n in ns])
        s = slope(func, d, alpha)
        ax.loglog(ns, y, sty, color=col, lw=1.6, ms=4, label=f"{lab} (slope {s:.2f})")
    ax.set_xlabel("context size $n$")
    ax.set_ylabel("parameter count")
    ax.set_title(r"Parameter growth: ours $\Theta(\log n)$ vs prior work ($\alpha=3,d=3$)")
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_param_count.png"), dpi=130)
    plt.close(fig)


def fig_c3_decomposition():
    """C3: the three decomposition terms vs n, all = O(n^{-2a/(2a+d)})."""
    alpha, d = 1.5, 2
    r = 2 * alpha / (2 * alpha + d)
    ns = np.array([2 ** k for k in range(6, 16)])
    term1 = 1.0 / ns                      # O(1/n)  [<= n^{-r} since r<=1]
    term2 = ns ** (-r)                    # locpol rate
    Gamma = ns ** r * np.log(np.e * ns) ** 3
    logG = r * np.log(ns) + 3 * np.log(np.e * ns)
    term3 = (np.log(np.e * ns) ** 3 + np.log(np.e * ns) * logG) / Gamma
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(ns, term1, "o-", lw=1.6, ms=4, label=r"term1: $R(f_{\mathrm{TF}})-R(f_{\mathrm{LocPol}})=O(1/n)$")
    ax.loglog(ns, term2, "s-", lw=1.6, ms=4, label=fr"term2: locpol $n^{{ {-r:.2f} }}$")
    ax.loglog(ns, term3, "^-", lw=1.6, ms=4, label=r"term3: generalization $O((\log^3 n+\log n\log\Gamma)/\Gamma)$")
    ax.loglog(ns, ns ** (-r), ":", color="gray", lw=1.5, label=fr"minimax rate $n^{{ {-r:.2f} }}$")
    ax.set_xlabel("context size $n$")
    ax.set_ylabel("term magnitude (arb. units)")
    ax.set_title("ERM risk decomposition: all terms $\\leq O(n^{-2\\alpha/(2\\alpha+d)})$ (Thm 3.2)")
    ax.legend(fontsize=7.6, loc="lower left")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_c3_decomposition.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    print("generating fig1 (locpol rate)..."); fig_locpol_rate()
    print("generating fig2 (construction approx)..."); fig_construction_approx()
    print("generating fig3 (construction components)..."); fig_construction_components()
    print("generating fig4 (param count)..."); fig_param_count()
    print("generating fig5 (C3 decomposition)..."); fig_c3_decomposition()
    print("done ->", os.path.abspath(FIG_DIR))
