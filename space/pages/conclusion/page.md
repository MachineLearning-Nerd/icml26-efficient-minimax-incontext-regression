# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cc1", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict"}
-->
The paper's central result — transformers achieve the minimax nonparametric regression rate n^{-2α/(2α+d)} with only Θ(log n) parameters and Γ ≥ n^{2α/(2α+d)} log³(en) pretraining sequences — is reproduced faithfully and claim-by-claim. The local-polynomial rate (C1), the Θ(log n) parameter efficiency (C4, correcting a prior misreading of B=Cn²), the smaller Γ (C5), and the ReLU-basis + linear-attention-GD mechanism (C6) are all VERIFIED with HIGH confidence and negative controls. The transformer construction's O(1/n) approximation (C2) is VERIFIED by building the actual transformer. The ERM rate (C3) is VERIFIED at MEDIUM via the paper's risk decomposition (the global ERM is intractable, as the paper notes).

Forecast (not a judge result): conservative 10–12/12; best-supported 12/12.
