# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "m1", "created_at": "2026-07-25T00:00:00+00:00", "title": "Implementation"}
-->
Faithful NumPy reconstruction of the paper's exact definitions:
- `repro/locpol.py`: the truncated local polynomial estimator (Theorem 2.5) — degree p=⌈α⌉, kernel K=(1−‖x‖₁)₊², bandwidth h=n^{-1/(2α+d)}, weighted least squares in the monomial basis. Multivariate.
- `repro/transformer.py`: the **exact** Definitions 2.1–2.4 — linear attention `Z + ZQ(ZK)ᵀZV`, ReLU FFN, blocks, full transformer.
- `repro/construction.py`: sets concrete weights so a transformer of that exact form approximates local polynomial regression (Section 4): exact ReLU kernel, ReLU monomial basis, linear-attention gradient descent.
- `repro/architecture.py`: the exact architecture constants (δ, d_ffn, D, L, B) and parameter counter.

Author code exists (`tianyima2000/ICL_LocPol`, GPU PyTorch, 50k AdamW steps on A100); it is **not** reproduced because GPU is out of scope here. Instead, the theoretical claims are verified by faithful construction + rate measurement + the paper's own risk-decomposition route (the global ERM is intractable and the paper states training dynamics are out of scope).


---
<!-- trackio-cell
{"type": "markdown", "id": "m2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Run"}
-->
**Fixed command (every node):** `bash repro/ci.sh   # -> uv sync --frozen && uv run python -m repro.run`

**Pinned env:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)

**Source:** https://github.com/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric (branch `orx/erm-rate-c3` @ `7b5707f`, mirrored to `main`).
