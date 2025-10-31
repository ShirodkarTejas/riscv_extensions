import argparse
import csv
import time
import math
from typing import Dict

import numpy as np

from runtime.api.python import sparse_attention

try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


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


def eval_once(B, H, L, D, pattern: str, params: Dict, precision: str, device: str, seed: int):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)
    if device == "cuda":
        assert HAS_TORCH, "PyTorch not available for CUDA runs"
        Qd = torch.as_tensor(Q, device="cuda")
        Kd = torch.as_tensor(K, device="cuda")
        Vd = torch.as_tensor(V, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Os = sparse_attention(Qd, Kd, Vd, pattern=pattern, params=params, precision=precision, training=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        O_sparse = Os.detach().cpu().float().numpy()
    else:
        t0 = time.perf_counter()
        O_sparse = sparse_attention(Q, K, V, pattern=pattern, params=params, precision=precision, training=False).astype(np.float32)
        t1 = time.perf_counter()

    # dense baseline for small L to estimate accuracy deltas
    acc_delta = None
    if L <= 256:
        O_dense = dense_attention(Q, K, V)
        acc_delta = float(np.mean(np.abs(O_dense - O_sparse)))

    latency_ms = (t1 - t0) * 1e3
    return latency_ms, acc_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="*", default=[2048, 8192, 32768])
    ap.add_argument("--pattern", type=str, choices=["block_topk", "sliding_global"], default="block_topk")
    ap.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--keep_ratio", type=float, default=0.12)
    ap.add_argument("--window_size", type=int, default=512)
    ap.add_argument("--global_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default="bench/results/eval_summary.csv")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    with open(args.csv, "w", newline="") as f:
        fieldnames = [
            "pattern", "precision", "device", "B", "H", "D", "L",
            "latency_ms", "acc_delta", "block_size", "keep_ratio", "window_size", "global_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for L in args.lengths:
            if args.pattern == "block_topk":
                params = {"block_size": args.block_size, "keep_ratio": args.keep_ratio, "global_tokens": args.global_tokens}
            else:
                params = {"window_size": args.window_size, "global_tokens": args.global_tokens, "block_size": args.block_size}
            lat, delta = eval_once(
                args.B, args.H, L, args.D, args.pattern, params, args.precision, args.device, args.seed
            )
            writer.writerow({
                "pattern": args.pattern,
                "precision": args.precision,
                "device": args.device,
                "B": args.B, "H": args.H, "D": args.D, "L": L,
                "latency_ms": lat,
                "acc_delta": delta,
                "block_size": args.block_size,
                "keep_ratio": args.keep_ratio,
                "window_size": args.window_size,
                "global_tokens": args.global_tokens,
            })
    print(f"Wrote summary to {args.csv}")


if __name__ == "__main__":
    main()


