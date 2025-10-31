import math
import numpy as np

from runtime.api.python import sparse_attention


def dense_attention(Q, K, V):
    B, H, L, D = Q.shape
    O = np.zeros_like(Q, dtype=np.float32)
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for h in range(H):
            Qbh = Q[b, h].astype(np.float32)  # [L, D]
            Kbh = K[b, h].astype(np.float32)
            Vbh = V[b, h].astype(np.float32)
            scores = Qbh @ Kbh.T * scale  # [L, L]
            # stable softmax per row
            m = scores.max(axis=1, keepdims=True)
            P = np.exp(scores - m)
            P /= P.sum(axis=1, keepdims=True) + 1e-12
            O[b, h] = P @ Vbh
    return O


def test_sparse_attention_block_topk_matches_dense_small():
    rng = np.random.default_rng(0)
    B, H, L, D = 1, 2, 64, 32
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)

    O_dense = dense_attention(Q, K, V)
    O_sparse = sparse_attention(
        Q, K, V,
        pattern="block_topk",
        params={"block_size": 16, "keep_ratio": 1.0, "global_tokens": 0},
        precision="bf16",
        training=False,
    ).astype(np.float32)

    max_delta = np.max(np.abs(O_dense - O_sparse))
    assert max_delta < 1e-4


def test_sparse_attention_sliding_global_degrades_reasonably():
    rng = np.random.default_rng(1)
    B, H, L, D = 1, 2, 96, 32
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)

    O_dense = dense_attention(Q, K, V)
    O_sparse = sparse_attention(
        Q, K, V,
        pattern="sliding_global",
        params={"window_size": 8, "global_tokens": 4, "block_size": 16},
        precision="bf16",
        training=False,
    ).astype(np.float32)

    # sanity: not catastrophically far from dense (tunable)
    mean_delta = float(np.mean(np.abs(O_dense - O_sparse)))
    assert mean_delta < 0.5


