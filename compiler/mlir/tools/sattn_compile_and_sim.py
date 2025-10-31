#!/usr/bin/env python3
import argparse
import os
import subprocess


def run(cmd, cwd=None):
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description="Run MLIR passes, emit indices, and run RoCC sim")
    ap.add_argument('--mlir', required=True, help='Input MLIR file')
    ap.add_argument('--indices', default='indices.txt')
    ap.add_argument('--sattn-opt', default='compiler/mlir/tools/sattn-opt/sattn-opt')
    ap.add_argument('--sim', default='hw/sim/obj_dir/Vrocc_sattn')
    ap.add_argument('--python', default='python')
    args = ap.parse_args()

    # 1) Run passes: materialize-indices, add tiling and fused softmax tags, and annotate lower-to-rocc
    out_mlir = os.path.splitext(args.mlir)[0] + '.lowered.mlir'
    run([args.sattn-opt, args.mlir, '-sattn-materialize-indices', '-sattn-tile', '-sattn-fuse-softmax', '-sattn-lower-to-rocc', '-o', out_mlir])

    # 2) Emit indices.txt
    run([args.python, 'compiler/mlir/tools/sattn_emit_indices_txt.py', '--in-mlir', out_mlir, '--out-indices', args.indices])

    # 3) Run sim
    run([args.sim, args.indices])


if __name__ == '__main__':
    main()


