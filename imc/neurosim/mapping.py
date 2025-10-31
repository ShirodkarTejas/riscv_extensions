import argparse
import csv
import math
from dataclasses import dataclass


@dataclass
class NeuroSimParams:
    array_rows: int = 256
    array_cols: int = 256
    adc_bits: int = 6
    tech_nm: int = 28


def active_tokens(pattern: str, L: int, block_size: int, keep_ratio: float, window_size: int, global_tokens: int) -> int:
    if pattern == "block_topk":
        num_blocks = math.ceil(L / block_size)
        k_blocks = max(1, int(math.ceil(keep_ratio * num_blocks)))
        return min(L, k_blocks * block_size + global_tokens)
    if pattern == "sliding_global":
        span = min(L, 2 * window_size + 1)
        return min(L, span + global_tokens)
    return L


def estimate_energy_latency(pattern: str, L: int, D: int, block_size: int, keep_ratio: float, window_size: int, global_tokens: int, p: NeuroSimParams):
    # Simple proxy: scale dense energy/latency by sparsity and add gather overhead
    S = active_tokens(pattern, L, block_size, keep_ratio, window_size, global_tokens)
    sparsity = S / max(1, L)
    dense_mac = L * D  # per row for QK^T or AV; proxy
    energy_dense = dense_mac  # normalized units
    latency_dense = dense_mac / (p.array_rows * p.array_cols)
    gather_overhead = 0.1 * energy_dense * sparsity
    energy_sparse = energy_dense * sparsity + gather_overhead
    latency_sparse = latency_dense * sparsity + 0.05 * latency_dense
    return {
        "S": S,
        "sparsity": sparsity,
        "energy_dense": energy_dense,
        "energy_sparse": energy_sparse,
        "latency_dense": latency_dense,
        "latency_sparse": latency_sparse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["block_topk", "sliding_global"], default="block_topk")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--keep_ratio", type=float, default=0.12)
    ap.add_argument("--window_size", type=int, default=512)
    ap.add_argument("--global_tokens", type=int, default=16)
    ap.add_argument("--csv", type=str, default="imc/neurosim/results_sparse_vs_dense.csv")
    args = ap.parse_args()

    p = NeuroSimParams()
    res = estimate_energy_latency(
        args.pattern, args.L, args.D, args.block_size, args.keep_ratio, args.window_size, args.global_tokens, p
    )

    import os
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pattern","L","D","block_size","keep_ratio","window_size","global_tokens",
            "S","sparsity","energy_dense","energy_sparse","latency_dense","latency_sparse",
        ])
        writer.writeheader()
        row = {**vars(args), **res}
        writer.writerow(row)
    print(f"Wrote IMC sparse vs dense proxy to {args.csv}")


if __name__ == "__main__":
    main()


