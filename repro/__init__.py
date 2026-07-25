"""CPU-only, SHA-bound reproduction of arXiv 2601.15014.

In-context nonparametric regression with transformers. Every numerical claim is
checked by a dedicated module under ``repro.claims``; ``repro.run`` is the single
fixed entry point that runs all of them and writes machine-readable evidence.
"""

__version__ = "0.1.0"
