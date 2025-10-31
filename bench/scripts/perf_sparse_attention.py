import argparse
import csv
import time
import math
import numpy as np

from runtime.api.python import sparse_attention
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def run_once(B, H, L, D, pattern, params, precision, seed, device="cpu"):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, L, D), dtype=np.float32)
    K = rng.standard_normal((B, H, L, D), dtype=np.float32)
    V = rng.standard_normal((B, H, L, D), dtype=np.float32)
    if device == "cuda":
        assert _HAS_TORCH, "PyTorch not available for CUDA runs"
        Q = torch.as_tensor(Q, device="cuda")
        K = torch.as_tensor(K, device="cuda")
        V = torch.as_tensor(V, device="cuda")
    t0 = time.perf_counter()
    sparse_attention(Q, K, V, pattern=pattern, params=params, precision=precision, training=False)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--pattern", type=str, choices=["block_topk", "sliding_global"], default="block_topk")
    ap.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_sizes", type=int, nargs="*", default=[32, 64, 128])
    ap.add_argument("--keep_ratios", type=float, nargs="*", default=[0.08, 0.12, 0.16, 0.24])
    ap.add_argument("--window_sizes", type=int, nargs="*", default=[256, 512, 1024])
    ap.add_argument("--global_tokens", type=int, nargs="*", default=[0, 8, 16, 32])
    ap.add_argument("--csv", type=str, default="-")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    rows = []
    if args.pattern == "block_topk":
        for bs in args.block_sizes:
            for kr in args.keep_ratios:
                for gt in args.global_tokens:
                    params = {"block_size": bs, "keep_ratio": kr, "global_tokens": gt}
                    # warmup
                    for _ in range(args.warmup):
                        run_once(args.B, args.H, args.L, args.D, args.pattern, params, args.precision, args.seed, args.device)
                    # repeats
                    times = [run_once(args.B, args.H, args.L, args.D, args.pattern, params, args.precision, args.seed + i, args.device) for i in range(args.repeat)]
                    rows.append({
                        "pattern": args.pattern,
                        "block_size": bs,
                        "keep_ratio": kr,
                        "global_tokens": gt,
                        "latency_ms": sum(times) / len(times),
                        "B": args.B, "H": args.H, "L": args.L, "D": args.D,
                        "precision": args.precision,
                        "device": args.device,
                    })
    else:
        for ws in args.window_sizes:
            for gt in args.global_tokens:
                for bs in args.block_sizes:
                    params = {"window_size": ws, "global_tokens": gt, "block_size": bs}
                    for _ in range(args.warmup):
                        run_once(args.B, args.H, args.L, args.D, args.pattern, params, args.precision, args.seed, args.device)
                    times = [run_once(args.B, args.H, args.L, args.D, args.pattern, params, args.precision, args.seed + i, args.device) for i in range(args.repeat)]
                    rows.append({
                        "pattern": args.pattern,
                        "window_size": ws,
                        "global_tokens": gt,
                        "block_size": bs,
                        "latency_ms": sum(times) / len(times),
                        "B": args.B, "H": args.H, "L": args.L, "D": args.D,
                        "precision": args.precision,
                        "device": args.device,
                    })

    if args.csv == "-":
        # stdout
        if rows:
            writer = csv.DictWriter(
                f := type("_StdOut", (), {"write": lambda self, s: print(s, end="")})(),
                fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    else:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


if __name__ == "__main__":
    main()


