# Branch audit

The repository accumulated five `orx/*` branches while the six claim routes
were developed. The final publication commit consolidated the work onto
`main`, so none of the remote branch tips is an ancestor of the final
publication commit. This is a history-consolidation result, not lost science:
tree comparisons found no unique source or verifier paths on the later three
branches, and the only branch-only paths on the other three were generated
`outputs/` artifacts that the final `.gitignore` intentionally regenerates.

| Historical branch | Role in the campaign | Disposition |
| --- | --- | --- |
| `orx/baseline` | Pin Python/uv, create the claim-result framework, and preserve the C0 toy/source-anchor control. | Framework and control superseded by the published `main`; generated outputs are reproducible. |
| `orx/symbolic-counting-c4-c5` | Add the architecture counter for C4 and the pretraining exponent comparison for C5. | Source modules and evidence are represented on `main`. |
| `orx/locpol-rate-c1` | Add the multivariate local-polynomial estimator, Hölder data, and C1 rate sweep. | Source modules and evidence are represented on `main`; generated C4/C5 outputs were branch-only. |
| `orx/construction-c6-c2` | Add the transformer construction, ReLU components, attention-GD check, and C2/C6 evidence. | Source modules and evidence are represented on `main`. |
| `orx/erm-rate-c3` | Add the C3 risk decomposition, covering-number argument, and empirical transformer witness. | Source modules and evidence are represented on `main`. |

The final public branch policy is deliberately simple: `main` is the only
remote branch. The old refs were removed after this audit; the root README and
committed `space/` evidence are the authoritative public surface.
