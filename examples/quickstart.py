import numpy as np

from runtime.api.python import sparse_attention


def run_cpu_block_topk():
    Q = np.random.randn(1, 2, 128, 64).astype(np.float32)
    K = np.random.randn(1, 2, 128, 64).astype(np.float32)
    V = np.random.randn(1, 2, 128, 64).astype(np.float32)
    O = sparse_attention(
        Q, K, V,
        pattern="block_topk",
        params={"block_size": 64, "keep_ratio": 0.12, "global_tokens": 16},
        precision="bf16",
        training=False,
    )
    print("CPU block_topk output shape:", O.shape)


def run_cuda_sliding_global():
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            print("CUDA not available; skipping GPU example")
            return
        Q = torch.randn(1, 2, 256, 64, device="cuda")
        K = torch.randn(1, 2, 256, 64, device="cuda")
        V = torch.randn(1, 2, 256, 64, device="cuda")
        O = sparse_attention(
            Q, K, V,
            pattern="sliding_global",
            params={"window_size": 32, "global_tokens": 8, "block_size": 64},
            precision="bf16",
            training=False,
        )
        print("CUDA sliding_global output shape:", tuple(O.shape))
    except Exception as e:
        print("GPU example failed:", e)


if __name__ == "__main__":
    run_cpu_block_topk()
    run_cuda_sliding_global()


