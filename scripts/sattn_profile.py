#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def run(cmd):
    print('[run]', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description='Profile sparse attention from MLIR on RVV or RoCC sim with selector flags')
    ap.add_argument('--mlir', required=True)
    ap.add_argument('--backend', choices=['rvv','sim'], default='rvv')
    ap.add_argument('--prefer-bsr', action='store_true')
    ap.add_argument('--prefer-sw', action='store_true')
    ap.add_argument('--l1-bytes', type=int, default=0)
    ap.add_argument('--use-hw-probe', action='store_true')
    ap.add_argument('--autotune', action='store_true')
    args = ap.parse_args()

    if args.backend == 'rvv':
        cmd = [sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', args.mlir]
        if args.prefer_bsr: cmd += ['--prefer-bsr']
        if args.prefer_sw: cmd += ['--prefer-sw']
        if args.l1_bytes: cmd += ['--l1-bytes', str(args.l1_bytes)]
        if args.use_hw_probe: cmd += ['--use-hw-probe']
        if args.autotune: cmd += ['--autotune']
        run(cmd)
    else:
        cmd = [sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', args.mlir]
        if args.prefer_bsr: cmd += ['--prefer-bsr']
        if args.prefer_sw: cmd += ['--prefer-sw']
        if args.l1_bytes: cmd += ['--l1-bytes', str(args.l1_bytes)]
        if args.use_hw_probe: cmd += ['--use-hw-probe']
        run(cmd)


if __name__ == '__main__':
    main()


