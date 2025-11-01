#!/usr/bin/env python3
import argparse
import subprocess


def run(cmd):
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def parse_metrics(line):
    vals = {}
    for tok in line.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            vals[k] = v
    return vals


def main():
    ap = argparse.ArgumentParser(description='Sweep GQA/compression settings and report RVV counters')
    ap.add_argument('--runner', default='build/backends/rvv/sattn_rvv_runner')
    ap.add_argument('--L', type=int, default=128)
    ap.add_argument('--D', type=int, default=32)
    ap.add_argument('--block_size', type=int, default=16)
    ap.add_argument('--keep_x1000', type=int, default=120)
    args = ap.parse_args()

    rows = []
    for gqa in [1, 2, 4]:
        for comp in [0, 8, 16]:
            cmd = [args.runner, '--spec', 'block_local_global', '--L', str(args.L), '--D', str(args.D),
                   '--block_size', str(args.block_size), '--keep_x1000', str(args.keep_x1000),
                   '--gqa_group_size', str(gqa), '--comp_block_size', str(comp)]
            line = run(cmd)
            m = parse_metrics(line)
            rows.append((gqa, comp, m.get('rvv_bytes_read','0'), m.get('bytes_written','0'), m.get('mac_flops','0')))

    print('| gqa_group_size | comp_block_size | bytes_read | bytes_written | mac_flops |')
    print('|---|---|---|---|---|')
    for gqa, comp, br, bw, mf in rows:
        print(f'| {gqa} | {comp} | {br} | {bw} | {mf} |')


if __name__ == '__main__':
    main()


