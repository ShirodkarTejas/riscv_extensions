import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from runtime.api.python import sparse_attention

try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@dataclass
class SearchSpace:
    block_sizes: list
    keep_ratios: list
    window_sizes: list
    global_tokens: list


def benchmark_once(B, H, L, D, pattern, params, precision, seed, device):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)
    if device == "cuda":
        assert HAS_TORCH, "PyTorch not available for CUDA runs"
        Q = torch.as_tensor(Q, device="cuda")
        K = torch.as_tensor(K, device="cuda")
        V = torch.as_tensor(V, device="cuda")
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    sparse_attention(Q, K, V, pattern=pattern, params=params, precision=precision, training=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3


def autotune_for_length(B, H, L, D, pattern, precision, space: SearchSpace, device, warmup=2, repeat=5, seed=0):
    best = {"latency_ms": float("inf"), "params": None}
    if pattern == "block_topk":
        for bs in space.block_sizes:
            for kr in space.keep_ratios:
                for gt in space.global_tokens:
                    params = {"block_size": bs, "keep_ratio": kr, "global_tokens": gt}
                    for _ in range(warmup):
                        benchmark_once(B, H, L, D, pattern, params, precision, seed, device)
                    times = [benchmark_once(B, H, L, D, pattern, params, precision, seed + i, device) for i in range(repeat)]
                    avg = sum(times) / len(times)
                    if avg < best["latency_ms"]:
                        best = {"latency_ms": avg, "params": params}
    elif pattern == "sliding_global":
        for ws in space.window_sizes:
            for gt in space.global_tokens:
                for bs in space.block_sizes:
                    params = {"window_size": ws, "global_tokens": gt, "block_size": bs}
                    for _ in range(warmup):
                        benchmark_once(B, H, L, D, pattern, params, precision, seed, device)
                    times = [benchmark_once(B, H, L, D, pattern, params, precision, seed + i, device) for i in range(repeat)]
                    avg = sum(times) / len(times)
                    if avg < best["latency_ms"]:
                        best = {"latency_ms": avg, "params": params}
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, choices=["block_topk", "sliding_global"], default="block_topk")
    ap.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="*", default=[2048, 8192, 32768])
    ap.add_argument("--block_sizes", type=int, nargs="*", default=[32, 64, 128])
    ap.add_argument("--keep_ratios", type=float, nargs="*", default=[0.08, 0.12, 0.16, 0.24])
    ap.add_argument("--window_sizes", type=int, nargs="*", default=[256, 512, 1024])
    ap.add_argument("--global_tokens", type=int, nargs="*", default=[0, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench/configs/best/best_configs.json")
    args = ap.parse_args()

    space = SearchSpace(
        block_sizes=args.block_sizes,
        keep_ratios=args.keep_ratios,
        window_sizes=args.window_sizes,
        global_tokens=args.global_tokens,
    )

    results = {"meta": {
        "pattern": args.pattern,
        "precision": args.precision,
        "device": args.device,
        "B": args.B, "H": args.H, "D": args.D,
        "timestamp_ms": int(time.time() * 1e3),
    }, "lengths": {}}

    for L in args.lengths:
        best = autotune_for_length(
            args.B, args.H, L, args.D, args.pattern, args.precision, space, args.device,
            warmup=args.warmup, repeat=args.repeat, seed=args.seed,
        )
        results["lengths"][str(L)] = best

    # ensure directory exists
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote best configs to {args.out}")


if __name__ == "__main__":
    main()


