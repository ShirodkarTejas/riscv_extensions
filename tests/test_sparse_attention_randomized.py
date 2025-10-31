import math
import numpy as np

from runtime.api.python import sparse_attention


def dense_attention(Q, K, V):
    B, H, L, D = Q.shape
    O = np.zeros_like(Q, dtype=np.float32)
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for h in range(H):
            Qbh = Q[b, h].astype(np.float32)
            Kbh = K[b, h].astype(np.float32)
            Vbh = V[b, h].astype(np.float32)
            scores = Qbh @ Kbh.T * scale
            m = scores.max(axis=1, keepdims=True)
            P = np.exp(scores - m)
            P /= P.sum(axis=1, keepdims=True) + 1e-12
            O[b, h] = P @ Vbh
    return O


def test_block_topk_equals_dense_when_keep1():
    rng = np.random.default_rng(123)
    for L in [32, 48, 64]:
        B, H, D = 1, 3, 32
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
        assert np.allclose(O_dense, O_sparse, atol=1e-4)


def test_determinism_same_inputs_same_outputs():
    rng = np.random.default_rng(7)
    B, H, L, D = 1, 2, 96, 32
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)
    params = {"block_size": 16, "keep_ratio": 0.12, "global_tokens": 8}
    O1 = sparse_attention(Q, K, V, pattern="block_topk", params=params, precision="bf16")
    O2 = sparse_attention(Q, K, V, pattern="block_topk", params=params, precision="bf16")
    assert np.array_equal(O1, O2)


def test_outputs_are_finite_and_reasonable_magnitude():
    rng = np.random.default_rng(17)
    B, H, L, D = 2, 2, 128, 32
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)
    O = sparse_attention(
        Q, K, V,
        pattern="sliding_global",
        params={"window_size": 16, "global_tokens": 4, "block_size": 16},
        precision="bf16",
        training=False,
    ).astype(np.float32)
    assert np.isfinite(O).all()
    assert np.max(np.abs(O)) < 1e3


