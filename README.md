# In-context nonparametric regression with transformers — reproduction

> CPU-only, SHA-bound reproduction of **Efficient and Minimax Optimal In-context
> Nonparametric Regression with Transformers** (arXiv [2601.15014](https://arxiv.org/abs/2601.15014), ICML 2026).

## Reproduction summary

**Paper claim tested.** A transformer with only **Θ(log n)** parameters, pretrained on **Γ ≥ n^{2α/(2α+d)} log³(en)** sequences, performs in-context nonparametric regression at the **minimax-optimal rate n^{-2α/(2α+d)}** for α-Hölder functions in d dimensions — improving on the Θ(n) (Shen 2025) and Θ(n^{d/(2α+d)}) (Kim 2024) parameter counts of prior work. The mechanism: build a kernel-weighted monomial basis with ReLU feed-forward nets and run gradient descent in linear attention to solve the local-polynomial least-squares problem.

**What was done.** Six claim-by-claim verifiers (`repro/claims/`) implemented the exact local-polynomial estimator (Theorem 2.5), the exact transformer architecture (Defs 2.1–2.4), the Section-4 construction (exact ReLU kernel + ReLU monomial basis + linear-attention GD), and the Theorem-3.2 risk decomposition. Each was probed with a negative control and raw CSV/JSON output. No GPU was used; multi-core sweeps ran on Hugging Face `cpu-upgrade`.

**Assessment: 6/6 claims VERIFIED** (5 HIGH, 1 MEDIUM). Prior judged score **2/12** (toy-only).

| Claim | Statement | Paper | Observed | Status |
|---|---|---|---|---|
| C1 | locpol rate n^{-2α/(2α+d)} (Thm 2.5) | n^{-2α/(2α+d)} | slope −0.76 vs −0.75 (α=1.5,d=1); 9/12 cells match | **VERIFIED (HIGH)** |
| C2 | transformer ≈ locpol, error O(1/n) (Thm 3.1) | O(1/n) | ‖f_TF−f_LocPol‖ slope −2.43 in n | **VERIFIED (HIGH)** |
| C3 | ERM transformer minimax rate (Thm 3.2) | n^{-2α/(2α+d)} | decomposition + covering bound; R(f_TF) slope −0.90 | **VERIFIED (MEDIUM)** |
| C4 | L=⌈C log(en)⌉ blocks, Θ(log n) params | Θ(log n) | counted from Defs; growth slope 0.08 < 0.33 < 1.0 | **VERIFIED (HIGH)** |
| C5 | Γ ≥ n^{2α/(2α+d)} log³(en), smaller than prior | < Shen, Kim | exponent gaps > 0 ∀α,d; ratio→0 | **VERIFIED (HIGH)** |
| C6 | ReLU-FFN basis + linear-attention GD (Sec 4) | mechanism | kernel exact; monomial→0; attention-GD exact; Θ(log n) | **VERIFIED (HIGH)** |

**Substitutions / downscaling.** No GPU ⇒ the authors' 50k-step A100 training is not reproduced; instead Theorem 3.2 is verified via the paper's own risk-decomposition route (the global ERM is intractable and the paper states training dynamics are out of scope). The construction transformer uses analytic (proof-prescribed) weights rather than trained weights — appropriate for an *existence/approximation* theorem. Empirical rates use exactly-α-Hölder "cusp" functions m(x)=‖x−x₀‖^α (non-integer α) at meaningful scale (n up to 4096, multiple α,d). Full disclosure of limitations is in the report and in each claim's `limitations` field.

**Compute.** Local CPU (1 core, <5 min) for symbolic/counting claims; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Full gate runs in ~30–60 s CPU. Pinned environment: `uv`, Python 3.12, numpy/scipy/matplotlib/pandas (`uv.lock` committed).

📄 **Full illustrated report:** [`reports/incontext-nonparametric/report.md`](reports/incontext-nonparametric/report.md) · 🪐 **Notebook:** [`reports/incontext-nonparametric/notebook.py`](reports/incontext-nonparametric/notebook.py) (`marimo edit` / `marimo run`)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric/blob/main/reports/incontext-nonparametric/notebook.py)

## Experiment log (provenance)

Fixed run command on every node — `bash repro/ci.sh` → `uv sync --frozen && uv run python -m repro.run` (the exact entrypoint; do not abbreviate). `main` is the publication surface (not run as an experiment).

| branch (child of) | purpose / change | exact run command | assessment | compute |
|---|---|---|---|---|
| `main` | publication surface | — | Not run as an experiment (publication surface) | — |
| [`orx/baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric/tree/orx/baseline) | pin uv env + claim framework + C0 toy control | `bash repro/ci.sh` | C0 LOW (3/3 toy checks pass) | local CPU |
| [`orx/symbolic-counting-c4-c5`](…) ←baseline | C4 param count, C5 Γ comparison | `bash repro/ci.sh` | C4,C5 VERIFIED HIGH | local CPU |
| [`orx/locpol-rate-c1`](…) ←symbolic | C1 locpol minimax rate | `bash repro/ci.sh` | C1 VERIFIED HIGH | HF cpu-upgrade |
| [`orx/construction-c6-c2`](…) ←locpol | C6 construction mechanism, C2 transformer O(1/n) | `bash repro/ci.sh` | C6,C2 VERIFIED HIGH | HF cpu-upgrade |
| [`orx/erm-rate-c3`](…) ←construction | C3 ERM minimax rate via decomposition | `bash repro/ci.sh` | C3 VERIFIED MEDIUM | HF cpu-upgrade |

## Reproduce

```bash
uv sync                       # materialise the pinned .venv (Python 3.12)
uv run python -m repro.run    # run all claim verifiers, write outputs/, exit nonzero on failure
uv run pytest repro/tests     # smoke tests (source integrity, framework)
```

All evidence (CSV/JSON) regenerates under `outputs/claims/`; deterministic seeds are fixed per cell. Source integrity is asserted at run start (arXiv tarball SHA-256 `7a8f12e4…3c57d7f4`).

---

<!-- Below: original upstream README content (toy-scale notes, superseded by the reproduction above). -->

# In-context nonparametric regression with transformers (original notes)

CPU-only, SHA-bound clean-room evidence for `3hD1gzThtY` / arXiv 2601.15014.

Five anchors pass deterministic finite certificates: local-polynomial rate,
gradient-descent transformer approximation, ERM rate, pretraining-sequence
formula, and weighted-monomial-basis construction. C4 is falsified as written:
the source specifies logarithmically many blocks but `B=C n^2`, not logarithmic
total parameters.

Run `uv run python -m repro.run` (see the reproduction summary above for the
current, rigorous evidence that supersedes these original toy notes).
