#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run RoCC Verilator sim from MLIR by emitting indices")
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--out", default="indices.txt")
    ap.add_argument("--sim", default="./obj_dir/Vrocc_sattn")
    ap.add_argument("--emit", default="python", help="python executable for emit script")
    args = ap.parse_args()

    # Emit indices
    cmd_emit = [args.emit, os.path.join("..", "..", "compiler", "mlir", "tools", "sattn_emit_indices_txt.py"),
                "--in-mlir", args.mlir, "--out-indices", args.out]
    print("[emit]", " ".join(cmd_emit))
    subprocess.check_call(cmd_emit, cwd=os.path.dirname(os.path.abspath(__file__)))

    # Run sim if available
    sim_path = args.sim
    if not os.path.isabs(sim_path):
        sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sim_path)
    if os.path.exists(sim_path):
        cmd_sim = [sim_path, args.out]
        print("[sim]", " ".join(cmd_sim))
        subprocess.check_call(cmd_sim)
    else:
        print(f"[warn] Simulator not found at {sim_path}. Indices written to {args.out}.")


if __name__ == "__main__":
    main()


