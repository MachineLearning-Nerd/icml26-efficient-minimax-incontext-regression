# ICML 2026 reproduction audit: efficient in-context nonparametric regression

Independent reproduction audit for *Efficient and Minimax Optimal In-context
Nonparametric Regression with Transformers* by Michelle Ching, Ioana Popescu,
Nico Smith, Tianyi Ma, William G. Underwood, and Richard J. Samworth.

[![Open the reproducible notebook in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-efficient-minimax-incontext-regression/blob/main/reports/incontext-nonparametric/notebook.py)

## Status

**Scoped reproduction status: 6/6 claim contracts pass.**

**Paper-level status: INCONCLUSIVE.**

The six claim verifiers pass their stated checks with five HIGH-confidence
results and one MEDIUM-confidence result. This is a faithful, executable
reproduction audit—not a foundational proof-assistant formalization and not a
reproduction of the authors' 50k-step A100 training run. `VERIFIED` below
means verified within the stated audit scope.

The audit is pinned to:

- Paper page: [arXiv:2601.15014](https://arxiv.org/abs/2601.15014) (current
  arXiv version v2)
- Local source archive: [`source/arxiv-2601.15014.tar`](source/arxiv-2601.15014.tar)
- Source SHA-256: `7a8f12e4f42513a87ce747648cecb00a0fac21a9979b3035c49c7b7b3c57d7f4`
- `paper.tex` SHA-256: `abbe2a52502fb09711c8295d665ff2e12338ca62106e2c666f422380e71b9331`
- `appendix.tex` SHA-256: `711afe3ec3b826dd219ead3cc97f1269352d237125c0bd94cc4902e9c1405839`
- Historical evaluator record: the earlier toy-only revision scored `2/12`;
  this repository does not claim a new judge score.

## What the paper studies

The paper proves that, for alpha-Hölder regression functions with `n`
in-context examples in dimension `d`, a pretrained transformer can achieve the
minimax mean-squared-error rate `n^(-2 alpha/(2 alpha+d))` using
`Theta(log n)` parameters and at least
`n^(2 alpha/(2 alpha+d)) log^3(en)` pretraining sequences. The construction
approximates a local-polynomial estimator with a kernel-weighted monomial
basis and gradient descent implemented inside linear attention.

## Claim ledger: what is checked and how

Every claim follows this evidence chain:

```text
pinned paper source → claim verifier → raw CSV/JSON evidence → negative control → gate
```

| Claim | Paper statement | Producer path and observed evidence | Scoped verdict and boundary |
| --- | --- | --- | --- |
| C1 | The truncated local-polynomial estimator achieves `n^(-2 alpha/(2 alpha+d))` (Theorem 2.5). | `repro/claims/c1_locpol_rate.py` with `repro/locpol.py` and `repro/data.py`; symbolic bias–variance balance, 12 `(alpha,d)` cells, `n=32..4096`, and a too-slow-bandwidth control. | **VERIFIED, HIGH.** The upper-rate evidence passes in all cells and 9/12 slopes match within tolerance. The matching minimax lower bound is inherited from classical results cited by the paper, not reproved here. |
| C2 | A transformer approximates local-polynomial regression with risk gap `O(1/n)` (Theorem 3.1). | `repro/claims/c2_transformer_approx.py` with `repro/construction.py`, `repro/transformer.py`, and `repro/locpol.py`; the constructed transformer's error slope is `-2.43`, with a shallow-construction control at about `-0.40`. | **VERIFIED, HIGH.** The explicit construction route passes. It is an analytic NumPy construction, not a trained transformer checkpoint. |
| C3 | The ERM transformer achieves the minimax rate (Theorem 3.2). | `repro/claims/c3_erm_rate.py` with `repro/risk.py`, C1/C2 outputs, parameter-Lipschitz and covering-number checks, risk-decomposition terms, and a concrete-transformer rate slope near `-0.90`. | **VERIFIED, MEDIUM.** The paper's decomposition is reconstructed and a class witness is measured; the global ERM is not solved exactly, consistent with the paper's stated scope. |
| C4 | Only `L=ceil(C log(en))` blocks and `Theta(log n)` total parameters are needed. | `repro/claims/c4_param_count.py` with `repro/architecture.py`; direct scalar counting from Definitions 2.1–2.4 and growth-rate comparison (`ours ~0.08`, Kim `~0.33`, Shen `1.0`). | **VERIFIED, HIGH.** This corrects the earlier misreading that `B=C n^2` was a parameter count; `B` is a per-entry magnitude bound. The prior baselines are used as stated in the paper. |
| C5 | The required pretraining sequence count is smaller: `Gamma >= n^(2 alpha/(2 alpha+d)) log^3(en)`. | `repro/claims/c5_pretraining.py` with `repro/architecture.py`; exponent-gap and asymptotic-ratio checks over 35 `(alpha,d)` cells. | **VERIFIED, HIGH.** The asymptotic formula comparison passes; the cited prior results are not independently re-derived from their original papers here. |
| C6 | ReLU FFNs build the kernel/basis and linear attention performs the GD construction (Section 4). | `repro/claims/c6_construction_components.py` with `repro/construction.py` and `repro/transformer.py`; exact kernel error 0, monomial approximation below `1e-3` at width 128, attention-GD max difference `2.7e-17`, and `Theta(log n)` step growth. | **VERIFIED, HIGH.** The mechanism is reproduced with explicit NumPy components; no author training dynamics or GPU checkpoint is claimed. |

The historical C0 module (`repro/claims/c0_baseline_reference.py`) preserves the
earlier toy/source-anchor control. It is not counted among the six paper claim
contracts. The generated claim records are written to `outputs/claims/` and
the committed Space mirror exposes the publication-facing CSV and figures.

## Repository map

- `repro/claims/` — one verifier per paper claim, including checks,
  confidence, source anchors, limitations, and negative controls.
- `repro/` — local-polynomial estimator, transformer architecture, analytic
  construction, risk decomposition, data generation, and the fixed entrypoint.
- `repro/ci.sh` — canonical environment-and-reproduction command.
- `repro/src/run_publication_gate.py` — wrapper that runs the current claim
  suite and writes the compact gate record.
- `source/` — the SHA-bound arXiv source archive.
- `reports/incontext-nonparametric/` — illustrated report and notebook.
- `space/` — evaluator-facing static evidence snapshot with claim pages,
  figures, and raw CSVs. The root README is authoritative for the current
  scope boundary.
- [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md) — exact role and disposition of every
  legacy `orx/*` branch.

## Reproduce

The fixed command uses Python 3.12 and the committed `uv.lock`:

```bash
bash repro/ci.sh
```

It runs `uv sync --frozen`, verifies the paper-source hashes, executes C1–C6
and the historical C0 control, writes `outputs/claims/*.json` and CSVs, and
fails if a verifier reports a failed check. A successful run also writes
[`outputs/publication_gate.json`](outputs/publication_gate.json).

The author implementation is noted in the report as
[`tianyima2000/ICL_LocPol`](https://github.com/tianyima2000/ICL_LocPol), but its
GPU PyTorch/A100 training is outside this audit. The reproduced paths instead
test the paper's analytic construction and risk-decomposition claims directly.

## Evidence limitations

- Finite rate sweeps corroborate the stated asymptotics; they do not establish
  universal quantifiers by themselves.
- C1 measures the sharp upper-bound behavior at a Hölder cusp. Its matching
  minimax lower bound is a cited classical result.
- C3 reconstructs the theorem's risk decomposition and supplies a concrete
  transformer witness, but does not compute the global ERM exactly.
- C4 and C5 compare formulas and growth exponents from the paper and cited
  baselines; the cited baseline theorems are not independently re-derived.
- C2/C6 use explicit analytic weights and NumPy components. They do not claim
  to reproduce the authors' training dynamics, optimizer, hardware, or model
  checkpoint.

## Branch policy and history

The research campaign used five `orx/*` branches, each representing a staged
claim addition. Their code and evidence were consolidated into the published
`main` surface. Before deletion, the branch audit compared every branch tree;
branch-only paths were generated `outputs/` files, which are reproducible and
ignored in the final publication tree. See [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md).

The repository was renamed from
`icml26-repro-3hD1gzThtY-incontext-nonparametric` to
`icml26-efficient-minimax-incontext-regression` so the public name describes
the paper rather than the internal experiment identifier.

## Citation

```bibtex
@inproceedings{ching2026efficient,
  title     = {Efficient and Minimax Optimal In-context Nonparametric Regression with Transformers},
  author    = {Michelle Ching and Ioana Popescu and Nico Smith and Tianyi Ma and William G. Underwood and Richard J. Samworth},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  eprint    = {2601.15014},
  archivePrefix = {arXiv}
}
```

## Thank you

Thank you to Michelle Ching, Ioana Popescu, Nico Smith, Tianyi Ma, William G.
Underwood, and Richard J. Samworth for making the paper and source available.
This repository is an independent reproduction audit, not the authors'
official implementation.
