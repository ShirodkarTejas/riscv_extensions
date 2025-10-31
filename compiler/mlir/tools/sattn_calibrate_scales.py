#!/usr/bin/env python3
import argparse, os, re, subprocess, sys


def run(cmd):
    print('[run]', ' '.join(cmd))
    return subprocess.check_output(cmd, text=True)


def parse_attrs(txt):
    m = re.search(r"sattn\.sparse_attention[^\{]*\{([^}]*)\}", txt, re.MULTILINE | re.DOTALL)
    if not m:
        return {}
    attrs = m.group(1)
    def geti(name):
        mm = re.search(rf"{name}\s*=\s*([0-9]+)", attrs)
        return int(mm.group(1)) if mm else None
    def gets(name):
        mm = re.search(rf"{name}\s*=\s*\"([^\"]+)\"", attrs)
        return mm.group(1) if mm else None
    def getf(name):
        mm = re.search(rf"{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)", attrs)
        return float(mm.group(1)) if mm else None
    return {
        'spec': gets('spec') or 'sliding_window',
        'L': geti('s_tokens') or geti('tile_S') or 128,
        'D': geti('head_dim_d') or geti('tile_D') or 32,
        'window_size': geti('window_size') or 8,
        'block_size': geti('block_size') or 64,
        'keep_ratio': getf('keep_ratio') or 0.12,
        'global_tokens': geti('global_tokens') or 0,
    }


def main():
    ap = argparse.ArgumentParser(description='Calibrate per-tensor scales for i8/i4 using RVV runner synthetic data')
    ap.add_argument('--mlir', required=True)
    ap.add_argument('--precision', choices=['i8','i4'], default='i8')
    ap.add_argument('--runner', default='build/backends/rvv/sattn_rvv_runner')
    args = ap.parse_args()

    txt = open(args.mlir).read()
    a = parse_attrs(txt)
    keep_x1000 = int(round((a['keep_ratio'] if a['keep_ratio'] is not None else 0.12) * 1000.0))
    cmd = [args.runner,
           '--spec', a['spec'], '--L', str(a['L']), '--D', str(a['D']),
           '--window', str(a['window_size']), '--block_size', str(a['block_size']),
           '--global_tokens', str(a['global_tokens']), '--keep_x1000', str(keep_x1000),
           '--precision', args.precision, '--calibrate']
    # Fallback runner path if missing
    if not os.path.exists(cmd[0]) and os.path.exists('build/rvv/sattn_rvv_runner'):
        cmd[0] = 'build/rvv/sattn_rvv_runner'
    out = run(cmd)
    print(out)


if __name__ == '__main__':
    main()


