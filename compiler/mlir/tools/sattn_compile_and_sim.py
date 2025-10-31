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
    ap.add_argument('--sattn-opt', default='build/mlir/tools/sattn-opt/sattn-opt')
    ap.add_argument('--sim', default='hw/sim/obj_dir/Vrocc_sattn')
    ap.add_argument('--python', default='/opt/venv/bin/python')
    args = ap.parse_args()

    # 1) Run passes: materialize-indices, add tiling and fused softmax tags, and annotate lower-to-rocc
    out_mlir = os.path.splitext(args.mlir)[0] + '.lowered.mlir'
    try:
        if not os.path.exists(args.sattn_opt):
            raise FileNotFoundError(args.sattn_opt)
        run([args.sattn_opt, args.mlir, '-sattn-materialize-indices', '-sattn-tile', '-sattn-fuse-softmax', '-sattn-lower-to-rocc', '-o', out_mlir])
    except Exception as e:
        print(f"[warn] sattn-opt unavailable or failed ({e}); using input MLIR directly")
        out_mlir = args.mlir

    # 2) Emit indices.txt
    run([args.python, 'compiler/mlir/tools/sattn_emit_indices_txt.py', '--in-mlir', out_mlir, '--out-indices', args.indices])
    # Also emit descriptor to pass sizes to sim
    desc_txt = os.path.splitext(args.indices)[0] + '.desc'
    run([args.python, 'compiler/mlir/tools/sattn_emit_desc_txt.py', '--in-mlir', out_mlir, '--out-desc', desc_txt])

    # 3) Emit a simple descriptor text file for the sim
    # desc_txt = os.path.splitext(args.indices)[0] + '.desc'
    # with open(out_mlir, 'r') as f:
    #     txt = f.read()
    # import re
    # m = re.search(r'sattn\.rocc_call[^\{]*\{([^}]*)\}', txt, re.MULTILINE | re.DOTALL)
    # if m:
    #     attrs = m.group(1)
    #     def get(name, default):
    #         mm = re.search(rf'{name}\s*=\s*([0-9]+)', attrs)
    #         return int(mm.group(1)) if mm else default
    #     M = get('m_rows', 4)
    #     D = get('head_dim_d', 16)
    #     S = get('s_tokens', 16)
    #     BS = get('block_size', 4)
    #     KB = get('k_blocks', max(1, (S + BS - 1)//BS))
    # else:
    #     M, D, S, BS, KB = 4, 16, 16, 4, 4
    # with open(desc_txt, 'w') as df:
    #     df.write(f"m_rows={M}\nhead_dim_d={D}\nblock_size={BS}\nk_blocks={KB}\ns_tokens={S}\n")

    # 4) Run sim (indices.txt and descriptor)
    run([args.sim, args.indices, desc_txt])


if __name__ == '__main__':
    main()


