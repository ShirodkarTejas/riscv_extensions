#!/usr/bin/env python3
import argparse
import math
import re


def parse_values(mlir_text: str):
    # Prefer lowered rocc_call
    m = re.search(r"sattn\.rocc_call[^\{]*\{([^}]*)\}", mlir_text, re.MULTILINE | re.DOTALL)
    if m:
        attrs = m.group(1)
        def geti(name, default):
            mm = re.search(rf"{name}\s*=\s*([0-9]+)", attrs)
            return int(mm.group(1)) if mm else default
        return {
            'm_rows': geti('m_rows', 4),
            'head_dim_d': geti('head_dim_d', 16),
            'block_size': geti('block_size', 4),
            's_tokens': geti('s_tokens', 16),
            'k_blocks': geti('k_blocks', 4),
            'global_tokens': geti('global_tokens', 0),
        }
    # Fallback from unlowered attrs
    # Try to isolate the attribute dict first (more robust across formatting)
    m2 = re.search(r"sattn\\.sparse_attention[^\{]*\{([^}]*)\}", mlir_text, re.MULTILINE | re.DOTALL)
    scope = m2.group(1) if m2 else mlir_text
    def find_int(name, default):
        mm = re.search(rf'{name}\\s*=\\s*([0-9]+)\\s*:\\s*i64', scope)
        return int(mm.group(1)) if mm else default
    S = find_int('tile_S', 16)
    BS = find_int('block_size', 4)
    return {
        'm_rows': 4,
        'head_dim_d': 16,
        'block_size': BS,
        's_tokens': S,
        'k_blocks': math.ceil(S / (BS if BS else 1)),
        'global_tokens': find_int('global_tokens', 0),
    }


def main():
    ap = argparse.ArgumentParser(description='Emit simple descriptor .desc from MLIR')
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--out-desc', required=True)
    args = ap.parse_args()
    txt = open(args.in_mlir).read()
    vals = parse_values(txt)
    with open(args.out_desc, 'w') as f:
        for k in ['m_rows', 'head_dim_d', 'block_size', 'k_blocks', 's_tokens', 'global_tokens']:
            f.write(f"{k}={vals.get(k, 0)}\n")


if __name__ == '__main__':
    main()


