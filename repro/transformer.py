"""Exact transformer architecture from the paper (Definitions 2.1-2.4).

Implemented in NumPy exactly as defined:
  - Linear attention layer (Def 2.1):  Attn(Z) = Z + Z Q (Z K)^T Z V
  - Feed-forward network layer (Def 2.2): FFN(Z) = Z + (W2 ReLU(W1 Z^T + b1 1^T) + b2 1^T)^T
  - Block (Def 2.3):  Block = FFN ∘ Attn
  - Transformer (eq. TF): TF(Z) = Block_L ∘ ... ∘ Block_1 (Z)

These ARE the paper's definitions (no softmax, no normalisation, single-head).
The construction module sets concrete weights so that a transformer of this exact
form approximates the local polynomial estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def linear_attention(Z: np.ndarray, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Def 2.1: Attn_{Q,K,V}(Z) = Z + Z Q (Z K)^T Z V.  Z: (n+1, delta)."""
    return Z + Z @ Q @ (Z @ K).T @ Z @ V


def ffn(Z: np.ndarray, W1: np.ndarray, W2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Def 2.2: FFN(Z) = Z + (W2 ReLU(W1 Z^T + b1 1^T) + b2 1^T)^T.  Applied per row."""
    # Z: (T, delta). W1: (dffn, delta), W2: (delta, dffn), b1: (dffn,), b2: (delta,).
    T = Z.shape[0]
    pre = Z @ W1.T + b1[None, :]          # (T, dffn)
    post = relu(pre) @ W2.T + b2[None, :]  # (T, delta)
    return Z + post


@dataclass
class BlockParams:
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    W1: np.ndarray
    W2: np.ndarray
    b1: np.ndarray
    b2: np.ndarray


def block(Z: np.ndarray, p: BlockParams) -> np.ndarray:
    """Def 2.3: Block = FFN ∘ Attn."""
    return ffn(linear_attention(Z, p.Q, p.K, p.V), p.W1, p.W2, p.b1, p.b2)


def transformer(Z: np.ndarray, blocks: list[BlockParams]) -> np.ndarray:
    """eq. (TF): compose L blocks."""
    for bp in blocks:
        Z = block(Z, bp)
    return Z


# --- a minimal ReLU-network helper for the construction (per-token FFNs) ----
@dataclass
class ReLUNet:
    """f(x) = W_L ReLU(...ReLU(W_1 x + b_1)...) + b_L. Width/depth arbitrary."""
    W: list[np.ndarray]  # weight matrices
    b: list[np.ndarray]  # biases

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (..., d_in). Apply across the last axis.
        z = x
        for i in range(len(self.W) - 1):
            z = relu(z @ self.W[i].T + self.b[i])
        return z @ self.W[-1].T + self.b[-1]

    @property
    def depth(self) -> int:
        return len(self.W) - 1

    def param_count(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))


def relu_net_from_affine_layers(layers: list[tuple[np.ndarray, np.ndarray, bool]]) -> ReLUNet:
    """Build a ReLUNet from (W, b, use_relu_after) affine layers."""
    W, b = [], []
    for M, off, _ in layers:
        W.append(M)
        b.append(off)
    return ReLUNet(W, b)
