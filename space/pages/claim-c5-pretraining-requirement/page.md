# Claim C5 — pretraining requirement


---
<!-- trackio-cell
{"type": "markdown", "id": "C5_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: HIGH**

Our exponent 2a/(2a+d) is strictly below Shen's (6a+d)/(2a+d) by (4a+d)/(2a+d) and below Kim's (2a+2d)/(2a+d) by 2d/(2a+d), for every alpha>0, d>=1. The polynomial exponent gap dominates the log factors, so Gamma_ours/Gamma_prior -> 0 as n grows (verified monotone). This is the exact asymptotic comparison the claim asserts.

**Exact claim tested.** Required pretraining sequences Gamma >= C n^{2 alpha/(2 alpha+d)} log^3(en), smaller than Omega(n^{(6a+d)/(2a+d)} log n) (Shen 2025) and Omega(n^{(2a+2d)/(2a+d)} log n) (Kim 2024).

| check | passed | detail |
|---|---|---|
| source_gamma_formula | PASS | Theorem 3.2 states Gamma >= C n^{2a/(2a+d)} log^3(en) |
| source_prior_requirements_stated | PASS | paper states Shen Omega(n^{(6a+d)/(2a+d)} log n) and Kim Omega(n^{(2a+2d)/(2a+d)} log n) |
| exponent_gaps_strictly_positive | PASS | For all 35 (alpha>0, d>=1) cells: Shen-ours gap=(4a+d)/(2a+d)>0, Kim-ours gap=2d/(2a+d)>0 |
| ratio_eventually_below_one_for_all_alpha_d | PASS | Gamma_ours/Gamma_prior crosses below 1 for finite n for every tested (alpha,d) |
| ratio_decreases_in_large_n_regime | PASS | Gamma_ours/Gamma_Shen at 2^48 < at 2^30 (polynomial gap dominates log factors) |
| ratio_tends_to_zero | PASS | Gamma_ours/Gamma_Kim shrinks >100x from 2^40 to 2^100 and <1e-2 at 2^100 (exponent gap>0 => n^{-g} log^2(en) -> 0) |



---
<!-- trackio-cell
{"type": "markdown", "id": "C5_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Theorem 3.2: Gamma >= C n^{2a/(2a+d)} log^3(en)
- paper.tex Sec 3: Shen Omega(n^{(6a+d)/(2a+d)} log n)
- paper.tex Sec 3: Kim Omega(n^{(2a+2d)/(2a+d)} log n)

**Executable verifier & fixed command:**
- Code: [`repro/claims/c5_pretraining.py`](https://github.com/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric/blob/orx/erm-rate-c3/repro/claims/c5_pretraining.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python -m repro.run`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `orx/erm-rate-c3` @ `7b5707f` (mirrored to `main`).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 0.002.

**Raw data (downloadable CSV):**

- [`evidence/pretraining_comparison.csv`](../evidence/pretraining_comparison.csv)
- [`evidence/exponent_gaps.csv`](../evidence/exponent_gaps.csv)

**Limitations & deviations:**

- Log factor differs (ours log^3(en) vs prior log n); for small n the ratio can be >1 transiently, but the claim is asymptotic and the polynomial exponent gap dominates. Prior exponents taken as stated in this paper.
