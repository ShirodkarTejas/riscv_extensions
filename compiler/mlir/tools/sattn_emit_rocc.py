#!/usr/bin/env python3
"""
Emit a simple RoCC descriptor JSON (indices + sizes) from a sattn MLIR file.
This is a lightweight parser that looks for attributes on sattn.sparse_attention.
"""
import argparse
import json
import re


def parse_mlir(path: str):
    with open(path, 'r') as f:
        txt = f.read()
    # Find first sattn.sparse_attention op block
    m = re.search(r'"sattn\.sparse_attention"\([^)]*\)\s*\{([^}]*)\}', txt, re.MULTILINE | re.DOTALL)
    if not m:
        raise SystemExit("No sattn.sparse_attention op found")
    attrs = m.group(1)
    def get_attr(name, typ=str, default=None):
        mm = re.search(rf'{name}\s*=\s*"?([A-Za-z0-9_\.]+)"?', attrs)
        if not mm:
            return default
        val = mm.group(1)
        if typ is int:
            return int(val)
        if typ is float:
            return float(val)
        return val
    pattern = get_attr('pattern', str, 'sliding_global')
    block_size = get_attr('block_size', int, 64)
    keep_ratio = get_attr('keep_ratio', float, 0.12)
    global_tokens = get_attr('global_tokens', int, 16)
    window_size = get_attr('window_size', int, 512)
    # tiling hints if present
    tile_M = get_attr('tile_M', int, 64)
    tile_D = get_attr('tile_D', int, 64)
    tile_S = get_attr('tile_S', int, 128)
    return {
        'pattern': pattern,
        'block_size': block_size,
        'keep_ratio': keep_ratio,
        'global_tokens': global_tokens,
        'window_size': window_size,
        'tile_M': tile_M,
        'tile_D': tile_D,
        'tile_S': tile_S,
    }


def make_indices(cfg):
    # For demo: produce a simple 0..S-1 index list; real path would compute BSR indices
    S = cfg.get('tile_S', 128)
    return list(range(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    cfg = parse_mlir(args.in_mlir)
    indices = make_indices(cfg)
    desc = {
        'm_rows': cfg['tile_M'],
        'head_dim_d': cfg['tile_D'],
        's_tokens': cfg['tile_S'],
        'indices': indices,
    }
    with open(args.out_json, 'w') as f:
        json.dump(desc, f, indent=2)
    print(f"Wrote RoCC descriptor to {args.out_json}")


if __name__ == '__main__':
    main()


