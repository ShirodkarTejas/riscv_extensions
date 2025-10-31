#!/usr/bin/env python3
"""
Emit a plain-text indices file (one integer per line) from a sattn MLIR file.
Useful for preloading the RoCC sim index RAM.
"""
import argparse
import re


def parse_mlir_attrs(path: str):
    with open(path, 'r') as f:
        txt = f.read()
    m = re.search(r'"sattn\.sparse_attention"\([^)]*\)\s*\{([^}]*)\}', txt, re.MULTILINE | re.DOTALL)
    if not m:
        raise SystemExit("No sattn.sparse_attention op found")
    attrs = m.group(1)
    def get(name, typ=str, default=None):
        mm = re.search(rf'{name}\s*=\s*"?([A-Za-z0-9_\.]+)"?', attrs)
        if not mm:
            return default
        val = mm.group(1)
        if typ is int:
            return int(val)
        if typ is float:
            return float(val)
        return val
    # Parse indices array if present
    midx = re.search(r'indices\s*=\s*\[([^\]]*)\]', attrs)
    indices = None
    if midx:
        raw = midx.group(1)
        nums = re.findall(r'-?\d+', raw)
        try:
            indices = [int(n) for n in nums]
        except Exception:
            indices = None
    midxb = re.search(r'block_indices\s*=\s*\[([^\]]*)\]', attrs)
    block_indices = None
    if midxb:
        raw = midxb.group(1)
        nums = re.findall(r'-?\d+', raw)
        try:
            block_indices = [int(n) for n in nums]
        except Exception:
            block_indices = None
    return {
        'pattern': get('pattern', str, 'sliding_global'),
        'tile_S': get('tile_S', int, 128),
        'window_size': get('window_size', int, 512),
        'block_size': get('block_size', int, 64),
        'keep_ratio': get('keep_ratio', float, 0.12),
        'indices': indices,
        'block_indices': block_indices,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--out-indices', required=True)
    args = ap.parse_args()

    cfg = parse_mlir_attrs(args.in_mlir)
    S = int(cfg['tile_S'])
    if cfg['pattern'] == 'block_topk' and cfg.get('block_indices'):
        # write block indices directly; gather expands to tokens
        idx = cfg['block_indices']
    elif cfg['indices']:
        idx = cfg['indices'][:S]
    elif cfg['pattern'] == 'sliding_global':
        w = int(cfg['window_size'])
        cnt = min(S, max(1, 2 * w + 1))
        idx = list(range(cnt))
        if len(idx) < S:
            idx += list(range(cnt, S))
    elif cfg['pattern'] == 'block_topk':
        bs = int(cfg['block_size']); kr = float(cfg['keep_ratio'])
        k_blocks = max(1, int((S + bs - 1) // bs * kr))
        total = min(S, k_blocks * bs)
        idx = list(range(total))
        if len(idx) < S:
            idx += list(range(total, S))
    else:
        idx = list(range(S))

    with open(args.out_indices, 'w') as f:
    for v in idx[:S]:
            f.write(f"{v}\n")
    print(f"Wrote indices to {args.out_indices} ({len(idx[:S])} entries)")


if __name__ == '__main__':
    main()


