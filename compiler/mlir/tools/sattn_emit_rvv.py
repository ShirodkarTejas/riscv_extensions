#!/usr/bin/env python3
"""
Emit RVV bench args from a sattn MLIR op and optionally run the CLI bench.
"""
import argparse
import os
import re
import subprocess


def parse_mlir(path: str):
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
    return {
        'pattern': get('pattern', str, 'sliding_global'),
        'tile_M': get('tile_M', int, 64),
        'tile_D': get('tile_D', int, 64),
        'tile_S': get('tile_S', int, 128),
        'window_size': get('window_size', int, 16),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-mlir', required=True)
    ap.add_argument('--bench', default='backends/rvv/build/sattn_rvv_bench_cli')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()

    cfg = parse_mlir(args.in_mlir)
    if cfg['pattern'] != 'sliding_global':
        print('[warn] RVV CLI bench helper supports sliding_global only; proceeding.')
    cmd = [args.bench, '--B', '1', '--H', '2', '--L', str(cfg['tile_S']), '--D', str(cfg['tile_D']), '--window', str(cfg['window_size'])]
    print(' '.join(cmd))
    if args.run:
        subprocess.check_call(cmd)


if __name__ == '__main__':
    main()


