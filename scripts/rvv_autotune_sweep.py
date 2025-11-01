#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd):
    print('[run]', ' '.join(cmd))
    return subprocess.check_output(cmd, text=True)


def parse_metrics(line: str):
    # Expect checksum=... rvv_bytes_read=... bytes_written=... mac_flops=... rvv_cycles=...
    fields = {}
    for tok in line.strip().split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            fields[k] = v
    return fields


def native_runner_path():
    p = 'build/backends/rvv/sattn_rvv_runner'
    if os.path.exists(p):
        return p
    alt = 'build/rvv/sattn_rvv_runner'
    return alt if os.path.exists(alt) else p


def qemu_prefix():
    return ['qemu-riscv64', '-L', '/usr/riscv64-linux-gnu', '-cpu', 'rv64,v=true,vlen=128,elen=64']


def ensure_qemu_build(build_dir: str):
    # Build once by delegating to the helper; ignores failure if missing toolchain
    try:
        run([sys.executable, 'scripts/build_and_run_rvv_qemu.py', '--build-dir', build_dir, '--exe', 'sattn_rvv_runner', '--args', '--spec sliding_window --L 8 --D 8 --window 1'])
    except subprocess.CalledProcessError:
        pass


def write_csv(path, results):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tile_rows', 'checksum', 'rvv_bytes_read', 'bytes_written', 'mac_flops', 'rvv_cycles'])
        for tr, m in results:
            w.writerow([
                tr,
                m.get('checksum',''), m.get('rvv_bytes_read',''), m.get('bytes_written',''),
                m.get('mac_flops',''), m.get('rvv_cycles','')
            ])


def write_markdown(path, results, title=None):
    lines = []
    if title:
        lines.append(f"### {title}")
    lines.append('| tile_rows | checksum | rvv_bytes_read | bytes_written | mac_flops | rvv_cycles |')
    lines.append('|---|---|---|---|---|---|')
    for tr, m in results:
        lines.append(f"| {tr} | {m.get('checksum','-')} | {m.get('rvv_bytes_read','-')} | {m.get('bytes_written','-')} | {m.get('mac_flops','-')} | {m.get('rvv_cycles','-')} |")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Sweep RVV tile_rows and report metrics (native or QEMU)')
    ap.add_argument('--spec', default='sliding_window', choices=['sliding_window','block_local_global','nm_structured','lsh'])
    ap.add_argument('--L', type=int, default=128)
    ap.add_argument('--D', type=int, default=32)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--block_size', type=int, default=64)
    ap.add_argument('--keep_x1000', type=int, default=120)
    ap.add_argument('--tiles', default='1,2,4,8,16,32')
    ap.add_argument('--qemu', action='store_true')
    ap.add_argument('--build-dir', default='build/rvv-riscv64')
    ap.add_argument('--autotune', action='store_true', help='Also run runner --autotune and print chosen tile_rows if available')
    ap.add_argument('--export-csv', default='', help='Export results to CSV path')
    ap.add_argument('--export-md', default='', help='Export results to Markdown table at path')
    ap.add_argument('--label', default='', help='Optional title for Markdown export')
    args = ap.parse_args()

    tiles = [int(t) for t in args.tiles.split(',') if t]
    base_cmd = ['--spec', args.spec, '--L', str(args.L), '--D', str(args.D)]
    if args.spec in ('sliding_window','block_local_global'):
        base_cmd += ['--window', str(args.window)]
    if args.spec in ('block_local_global','nm_structured'):
        base_cmd += ['--block_size', str(args.block_size), '--keep_x1000', str(args.keep_x1000)]

    results = []
    if args.qemu:
        if not shutil.which('qemu-riscv64') or not shutil.which('riscv64-linux-gnu-gcc'):
            print('[skip] QEMU or riscv toolchain not found')
            return
        ensure_qemu_build(args.build_dir)
        exe = os.path.join(args.build_dir, 'sattn_rvv_runner')
        if not os.path.exists(exe):
            print('[warn] QEMU build missing runner; aborting sweep')
            return
        for tr in tiles:
            cmd = qemu_prefix() + [exe] + base_cmd + ['--tile_rows', str(tr)]
            out = run(cmd)
            metrics = parse_metrics(out.splitlines()[-1]) if out.strip() else {}
            results.append((tr, metrics))
    else:
        exe = native_runner_path()
        if not os.path.exists(exe):
            print('[warn] Native runner not found; please build backends/rvv')
            return
        for tr in tiles:
            cmd = [exe] + base_cmd + ['--tile_rows', str(tr)]
            out = run(cmd)
            metrics = parse_metrics(out.splitlines()[-1]) if out.strip() else {}
            results.append((tr, metrics))

    # Print table
    print('\n=== Sweep results (tile_rows) ===')
    print('tile_rows  checksum        rvv_bytes_read  bytes_written  mac_flops  rvv_cycles')
    best = None
    for tr, m in results:
        cyc = int(m.get('rvv_cycles','0')) if m.get('rvv_cycles') else 0
        if best is None or (cyc > 0 and cyc < best[0]):
            best = (cyc, tr)
        print(f"{tr:<9} {m.get('checksum','-'):<14} {m.get('rvv_bytes_read','-'):<15} {m.get('bytes_written','-'):<13} {m.get('mac_flops','-'):<9} {m.get('rvv_cycles','-')}")

    if args.autotune:
        cmd = ([exe] if not args.qemu else (qemu_prefix() + [exe])) + base_cmd + ['--autotune']
        out = run(cmd)
        print('\n[autotune] output:')
        print(out)

    if best and best[0] > 0:
        print(f"\nBest tile_rows by rvv_cycles: {best[1]} (cycles={best[0]})")

    # Exports
    if args.export_csv:
        try:
            write_csv(args.export_csv, results)
            print(f"[export] wrote CSV: {args.export_csv}")
        except Exception as e:
            print(f"[warn] failed to write CSV: {e}")
    if args.export_md:
        try:
            write_markdown(args.export_md, results, title=args.label)
            print(f"[export] wrote Markdown: {args.export_md}")
        except Exception as e:
            print(f"[warn] failed to write Markdown: {e}")


if __name__ == '__main__':
    main()


