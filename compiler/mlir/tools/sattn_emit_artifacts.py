#!/usr/bin/env python3
import argparse, os, re, subprocess, sys


def run(cmd):
    print('[run]', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description='Emit indices.txt and indices.desc from MLIR (unified)')
    ap.add_argument('--mlir', required=True)
    ap.add_argument('--out-stem', help='output path stem (without extension). Defaults to <mlir>.indices')
    args = ap.parse_args()

    mlir_path = args.mlir
    stem = args.out_stem or (os.path.splitext(mlir_path)[0] + '.indices')
    indices_txt = stem + '.txt'
    desc_txt = stem + '.desc'

    # Always emit indices and desc using existing tools
    run([sys.executable, 'compiler/mlir/tools/sattn_emit_indices_txt.py', '--in-mlir', mlir_path, '--out-indices', indices_txt])
    run([sys.executable, 'compiler/mlir/tools/sattn_emit_desc_txt.py', '--in-mlir', mlir_path, '--out-desc', desc_txt])

    # Patch-in extra attributes (nm_n, nm_m, lsh_buckets, keep_ratio, gqa_group_size, comp_block_size) if present
    try:
        txt2 = open(mlir_path, 'r', encoding='utf-8').read()
        def find_int(name):
            m = re.search(rf"{name}\\s*=\\s*([0-9]+)", txt2)
            return int(m.group(1)) if m else None
        def find_float(name):
            m = re.search(rf"{name}\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)", txt2)
            return float(m.group(1)) if m else None
        updates = {}
        for key in ['nm_n','nm_m','lsh_buckets','gqa_group_size','comp_block_size']:
            v = find_int(key)
            if v is not None: updates[key] = str(v)
        kr = find_float('keep_ratio')
        if kr is not None: updates['keep_ratio'] = str(kr)
        if updates:
            vals = {}
            with open(desc_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        vals[k] = v
            vals.update(updates)
            with open(desc_txt, 'w', encoding='utf-8') as f:
                keys = ['m_rows','head_dim_d','block_size','k_blocks','s_tokens','global_tokens','nm_n','nm_m','lsh_buckets','keep_ratio','gqa_group_size','comp_block_size']
                for k in keys:
                    f.write(f"{k}={vals.get(k, '0')}\n")
    except Exception:
        pass

    print(f"artifacts: indices={indices_txt} desc={desc_txt}")


if __name__ == '__main__':
    main()


