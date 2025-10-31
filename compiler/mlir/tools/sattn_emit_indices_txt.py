#!/usr/bin/env python3
import argparse
import math
import re


def parse_attrs(mlir_text: str):
    # Try lowered form first: sattn.rocc_call{ ... s_tokens=.., block_size=.. }
    m = re.search(r"sattn\.rocc_call[^\{]*\{([^}]*)\}", mlir_text, re.MULTILINE | re.DOTALL)
    if m:
        attrs = m.group(1)
        def get_int(name, default):
            mm = re.search(rf"{name}\s*=\s*([0-9]+)", attrs)
            return int(mm.group(1)) if mm else default
        s_tokens = get_int('s_tokens', 16)
        block_size = get_int('block_size', 4)
        return s_tokens, block_size

    # Fallback: unlowered sattn.sparse_attention with tile_S and block_size
    mm = re.search(r'sattn\.sparse_attention[^\"]*block_size\s*=\s*([0-9]+)\s*:\s*i64', mlir_text)
    bs = int(mm.group(1)) if mm else 4
    mm2 = re.search(r'tile_S\s*=\s*([0-9]+)\s*:\s*i64', mlir_text)
    s_tok = int(mm2.group(1)) if mm2 else 16
    return s_tok, bs


def main():
    ap = argparse.ArgumentParser(description="Emit indices.txt from lowered or raw MLIR")
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--out-indices', required=True)
    args = ap.parse_args()

    with open(args.in_mlir, 'r') as f:
        txt = f.read()

    s_tokens, block_size = parse_attrs(txt)
    idx_count = max(1, math.ceil(s_tokens / (block_size if block_size else 1)))
    with open(args.out_indices, 'w') as out:
        for i in range(idx_count):
            out.write(f"{i}\n")


if __name__ == '__main__':
    main()


