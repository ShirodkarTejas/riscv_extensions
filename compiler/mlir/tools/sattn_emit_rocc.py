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
    # Parse indices array if present
    indices = None
    midx = re.search(r'indices\s*=\s*\[([^\]]*)\]', attrs)
    if midx:
        raw = midx.group(1)
        nums = re.findall(r'-?\d+', raw)
        try:
            indices = [int(n) for n in nums]
        except Exception:
            indices = None
    # Parse block_indices for block_topk
    block_indices = None
    midxb = re.search(r'block_indices\s*=\s*\[([^\]]*)\]', attrs)
    if midxb:
        raw = midxb.group(1)
        nums = re.findall(r'-?\d+', raw)
        try:
            block_indices = [int(n) for n in nums]
        except Exception:
            block_indices = None
    return {
        'pattern': pattern,
        'block_size': block_size,
        'keep_ratio': keep_ratio,
        'global_tokens': global_tokens,
        'window_size': window_size,
        'tile_M': tile_M,
        'tile_D': tile_D,
        'tile_S': tile_S,
        'indices': indices,
        'block_indices': block_indices,
    }


def make_indices(cfg):
    """Produce a simple token index list consistent with pattern and tiles.
    - sliding_global: windowed neighborhood (clipped), padded to S
    - block_topk: select k_blocks*block_size tokens from the start (placeholder for true top-k)
    """
    S = int(cfg.get('tile_S', 128))
    pat = cfg.get('pattern', 'sliding_global')
    if pat == 'sliding_global':
        w = int(cfg.get('window_size', 512))
        count = min(S, max(1, 2 * w + 1))
        base = list(range(count))
        if len(base) < S:
            base += list(range(count, S))
        return base[:S]
    if pat == 'block_topk':
        bs = int(cfg.get('block_size', 64))
        kr = float(cfg.get('keep_ratio', 0.12))
        k_blocks = max(1, int((S + bs - 1) // bs * kr))
        total = min(S, k_blocks * bs)
        idx = list(range(total))
        if len(idx) < S:
            idx += list(range(total, S))
        return idx[:S]
    return list(range(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    cfg = parse_mlir(args.in_mlir)
    # Prefer block_indices for block_topk; otherwise use token indices
    if cfg.get('pattern') == 'block_topk' and cfg.get('block_indices'):
        indices = cfg['block_indices']
    else:
        indices = cfg.get('indices') or make_indices(cfg)
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


