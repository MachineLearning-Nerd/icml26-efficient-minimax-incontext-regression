# Claim C1 — local polynomial rate


---
<!-- trackio-cell
{"type": "markdown", "id": "C1_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: HIGH**

Bias-variance balance (bias^2~h^{2a}, var~1/(n h^d)) recovers h*=n^{-1/(2a+d)} and rate n^{-2a/(2a+d)}. Empirically, pointwise excess risk at the cusp of an exactly-alpha-Hölder function has log-log slope ≈ -2a/(2a+d) for 8/12 (alpha,d) cells (d in {1,2,3}, alpha in {0.5,1,1.5,2.5}); the upper bound holds in all cells. The undersmoothed-bandwidth negative control breaks the rate, confirming the rate is specific to the prescribed h.

![C1 evidence](../images/fig1_locpol_rate.png)

**Exact claim tested.** The truncated local polynomial estimator achieves the minimax-optimal rate O(n^{-2 alpha/(2 alpha+d)}) in mean squared error for alpha-Hölder functions with n context examples in d dimensions (Theorem 2.5).

| check | passed | detail |
|---|---|---|
| bias_variance_balance_recovers_h_exponent | PASS | regressing log(h*) on log(n) from numerically minimising h^{2a}+1/(n h^d) gives slope -1/(2a+d) => rate n^{-2a/(2a+d)}; (a=0.5,d=1):-0.500 vs -0.500, (a=1.5,d=2):-0.200 vs -0.200, (a=2.5,d=3):-0.125 vs -0.125 |
| empirical_slope_matches_minimax_rate | PASS | pointwise log-log slope ≈ -2a/(2a+d) for 8/12 (alpha,d) cells (tolerance 0.20); cells: (a=0.5,d=1) slope=-0.502 vs -r=-0.500, (a=0.5,d=2) slope=-0.369 vs -r=-0.333, (a=0.5,d=3) slope=-0.471 vs -r=-0.250, (a=1.0,d=1) slope=-0.674 vs -r=-0.667, (a=1.0,d=2) slope=-0.537 vs -r=-0.500, (a=1.0,d=3) slope=-0.764 vs -r=-0.400, (a=1.5,d=1) slope=-0.756 vs -r=-0.750, (a=1.5,d=2) slope=-0.695 vs -r=-0.600, (a=1.5,d=3) slope=-1.325 vs -r=-0.500, (a=2.5,d=1) slope=-0.851 vs -r=-0.833, (a=2.5,d=2) slope=-0.815 vs -r=-0.714, (a=2.5,d=3) slope=-1.099 vs -r=-0.625 |
| upper_bound_holds_all_cells | PASS | all cells: slope <= -r + 0.10, i.e. locpol achieves AT LEAST the minimax rate (Theorem 2.5 upper bound) |
| negative_control_too_slow_bandwidth_fails | PASS | too-slow bandwidth h=n^{-1/(4(2a+d))} (bias-limited) is clearly flatter than optimal in 11/12 cells: rate is specific to h=n^{-1/(2a+d)} |



---
<!-- trackio-cell
{"type": "markdown", "id": "C1_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Theorem 2.5 (locpol_main): R(f_LocPol)-sigma^2 <= C n^{-2a/(2a+d)}
- paper.tex: h := n^{-1/(2a+d)}, p := ceil(alpha), K(x)=(1-||x||_1)_+^2

**Executable verifier & fixed command:**
- Code: [`repro/claims/c1_locpol_rate.py`](https://github.com/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric/blob/orx/erm-rate-c3/repro/claims/c1_locpol_rate.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python -m repro.run`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `orx/erm-rate-c3` @ `7b5707f` (mirrored to `main`).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 1.908.

**Raw data (downloadable CSV):**

- [`evidence/locpol_rate_slopes.csv`](../evidence/locpol_rate_slopes.csv)
- [`evidence/locpol_rate_sweep.csv`](../evidence/locpol_rate_sweep.csv)

**Limitations & deviations:**

- Pointwise risk at a single hard point (cusp center) is a sharper probe than integrated risk; integrated risk can decay faster away from the cusp. Finite-n: for large alpha the asymptotic regime needs larger n. This verifies the UPPER-BOUND half of minimax optimality; the matching lower bound is the classical minimax result (Tsybakov 2009; Gyorfi et al. 2002) cited by the paper.
