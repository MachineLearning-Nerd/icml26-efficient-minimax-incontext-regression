# Claim C6 — ReLU basis + attention GD


---
<!-- trackio-cell
{"type": "markdown", "id": "C6_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: HIGH**

All three named mechanisms are implemented and verified: (A) the kernel K=(1-||x||_1)_+^2 is piecewise linear and its sqrt is an EXACT ReLU FFN; (B) ReLU FFNs approximate the monomial basis x^k with error -> 0 as width grows; (C) one Def-2.1 linear-attention block computes the GD sufficient statistics exactly, and ceil(C log(en)) GD steps converge to the least-squares solution. These are the actual ReLU-FFN and linear-attention layers of Definitions 2.1-2.2.

![C6 evidence](../images/fig3_construction_components.png)

**Exact claim tested.** The transformer implements the estimator by constructing a kernel-weighted monomial basis via ReLU feed-forward networks and then running gradient descent steps within linear attention layers to solve the local polynomial least-squares problem.

| check | passed | detail |
|---|---|---|
| relu_ffn_constructs_kernel_exactly | PASS | ReLU FFN realises K^{1/2}(u)=(1-||u||_1)_+ EXACTLY (max err < 1e-12) for d in {1,2,3,5}: {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0} |
| relu_ffn_approximates_monomial_basis | PASS | ReLU FFN approximates x^k (k in {2,3,4}); error strictly decreases with width and < 1e-3 at width 128. E.g. x^3: w=8:5.3e-02, w=16:1.2e-02, w=32:3.0e-03, w=64:7.4e-04, w=128:1.8e-04 |
| attention_block_computes_gd_statistics | PASS | one Def-2.1 linear-attention block computes the GD sufficient statistics M=sum a_i a_i^T and b=sum a_i y_i EXACTLY (max diff 2.22e-16) |
| theta_log_n_gd_steps_converge_to_lstsq | PASS | normal matrix kappa=210 constant in n (Theorem 2.5 eigenvalue bound); GD steps to reach 1/n grow as log(en) (T_min ratio 1.8 vs log ratio 1.8); T_min/log(en)~[73.47, 62.49, 65.18, 75.02] => Theta(log n) steps |



---
<!-- trackio-cell
{"type": "markdown", "id": "C6_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Section 4: kernel-weighted monomial basis via ReLU FFNs
- paper.tex Section 4: gradient descent within linear attention layers
- paper.tex Def 2.1 (linear attention), Def 2.2 (ReLU FFN)
- appendix sec:transformer_construction

**Executable verifier & fixed command:**
- Code: [`repro/claims/c6_construction_components.py`](https://github.com/MachineLearning-Nerd/icml26-efficient-minimax-incontext-regression/blob/main/repro/claims/c6_construction_components.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python repro/src/run_publication_gate.py`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `main` (publication surface).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 0.011.

**Raw data (downloadable CSV):**

- [`evidence/monomial_approx.csv`](../evidence/monomial_approx.csv)
- [`evidence/gd_convergence.csv`](../evidence/gd_convergence.csv)

**Limitations & deviations:**

- Monomial approximation here uses a width-controlled PL-interpolation ReLU network (error ~ width^{-2}); the paper's proof uses the stronger exponential (log-depth) rate of Lu-Shen-Yang-Zhang. The claim verified is that ReLU FFNs DO construct the basis to arbitrary precision (the mechanism), not the exact log-depth constant.
