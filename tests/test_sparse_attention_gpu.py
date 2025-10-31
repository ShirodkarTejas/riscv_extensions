import numpy as np

import pytest

from runtime.api.python import sparse_attention


def has_torch_cuda():
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not has_torch_cuda(), reason="CUDA not available or torch missing")
def test_triton_sliding_global_matches_cpu_reference_small():
    import torch
    rng = np.random.default_rng(0)
    B, H, L, D = 1, 2, 96, 64
    Q_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V_np = rng.standard_normal((B, H, L, D), dtype=np.float32)

    # CPU reference (numpy)
    O_cpu = sparse_attention(
        Q_np, K_np, V_np,
        pattern="sliding_global",
        params={"window_size": 8, "global_tokens": 4, "block_size": 16},
        precision="bf16",
        training=False,
    ).astype(np.float32)

    # Triton GPU
    Q_t = torch.as_tensor(Q_np, device="cuda")
    K_t = torch.as_tensor(K_np, device="cuda")
    V_t = torch.as_tensor(V_np, device="cuda")
    O_gpu = sparse_attention(
        Q_t, K_t, V_t,
        pattern="sliding_global",
        params={"window_size": 8, "global_tokens": 4, "block_size": 16},
        precision="bf16",
        training=False,
    ).cpu().float().numpy()

    # Tight tolerance for small shapes
    np.testing.assert_allclose(O_cpu, O_gpu, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not has_torch_cuda(), reason="CUDA not available or torch missing")
def test_block_topk_gpu_matches_dense_when_keep1():
    import torch
    rng = np.random.default_rng(2)
    B, H, L, D = 1, 2, 64, 32
    Q_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V_np = rng.standard_normal((B, H, L, D), dtype=np.float32)

    # CPU dense via keep_ratio=1.0 behavior of block_topk reference
    from tests.test_sparse_attention_cpu import dense_attention
    O_dense = dense_attention(Q_np, K_np, V_np)

    Q_t = torch.as_tensor(Q_np, device="cuda")
    K_t = torch.as_tensor(K_np, device="cuda")
    V_t = torch.as_tensor(V_np, device="cuda")
    O_gpu = sparse_attention(
        Q_t, K_t, V_t,
        pattern="block_topk",
        params={"block_size": 16, "keep_ratio": 1.0, "global_tokens": 0},
        precision="bf16",
        training=False,
    ).cpu().float().numpy()

    np.testing.assert_allclose(O_dense, O_gpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not has_torch_cuda(), reason="CUDA not available or torch missing")
def test_block_topk_triton_vs_cpu_reference_medium():
    import torch
    rng = np.random.default_rng(4)
    B, H, L, D = 1, 2, 96, 64
    Q_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V_np = rng.standard_normal((B, H, L, D), dtype=np.float32)

    # CPU reference sparse (python ref selects per-row blocks)
    O_cpu = sparse_attention(
        Q_np, K_np, V_np,
        pattern="block_topk",
        params={"block_size": 16, "keep_ratio": 0.5, "global_tokens": 4},
        precision="bf16",
        training=False,
    ).astype(np.float32)

    Q_t = torch.as_tensor(Q_np, device="cuda")
    K_t = torch.as_tensor(K_np, device="cuda")
    V_t = torch.as_tensor(V_np, device="cuda")
    O_gpu = sparse_attention(
        Q_t, K_t, V_t,
        pattern="block_topk",
        params={"block_size": 16, "keep_ratio": 0.5, "global_tokens": 4},
        precision="bf16",
        training=False,
    ).cpu().float().numpy()

    np.testing.assert_allclose(O_cpu, O_gpu, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not has_torch_cuda(), reason="CUDA not available or torch missing")
def test_triton_sliding_global_varied_dims_and_window():
    import torch
    rng = np.random.default_rng(3)
    # D not multiple of typical block to exercise masking
    B, H, L, D = 1, 1, 128, 80
    Q_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K_np = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V_np = rng.standard_normal((B, H, L, D), dtype=np.float32)

    for window in [4, 16, 32]:
        O_cpu = sparse_attention(
            Q_np, K_np, V_np,
            pattern="sliding_global",
            params={"window_size": window, "global_tokens": 0, "block_size": 16},
            precision="bf16",
            training=False,
        ).astype(np.float32)

        Q_t = torch.as_tensor(Q_np, device="cuda")
        K_t = torch.as_tensor(K_np, device="cuda")
        V_t = torch.as_tensor(V_np, device="cuda")
        O_gpu = sparse_attention(
            Q_t, K_t, V_t,
            pattern="sliding_global",
            params={"window_size": window, "global_tokens": 0, "block_size": 16},
            precision="bf16",
            training=False,
        ).cpu().float().numpy()

        np.testing.assert_allclose(O_cpu, O_gpu, atol=2e-3, rtol=2e-3)


