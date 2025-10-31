#!/usr/bin/env python3
import os
import re
import subprocess


def run(cmd):
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def parse_line(s):
    # spec=... checksum=... rvv_bytes_read=X bytes_written=Y mac_flops=Z
    parts = s.split()
    vals = {}
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            vals[k] = v
    return vals


def snapshot():
    exe = 'build/backends/rvv/sattn_rvv_runner'
    if not os.path.exists(exe):
        raise SystemExit(f"runner not found: {exe}")
    cases = [
        ('sliding_window', ['--L','128','--D','32','--window','8']),
        ('sliding_window i8', ['--L','128','--D','32','--window','8','--precision','i8','--scale_q_x1000','50','--scale_k_x1000','50','--scale_v_x1000','50']),
        ('block_local_global', ['--L','128','--D','32','--block_size','16','--keep_x1000','120','--global_tokens','4']),
        ('block_local_global i4', ['--L','128','--D','32','--block_size','16','--keep_x1000','120','--global_tokens','4','--precision','i4','--scale_q_x1000','125','--scale_k_x1000','125','--scale_v_x1000','125']),
    ]
    rows = []
    for name, args in cases:
        s = run([exe, '--spec', name.split()[0]] + args)
        vals = parse_line(s)
        rows.append((name, '128', '32', vals.get('rvv_bytes_read','0'), vals.get('bytes_written','0'), vals.get('mac_flops','0')))
    return rows


def update_readme(rows):
    p = 'backends/rvv/README.md'
    txt = open(p, 'r', encoding='utf-8').read()
    start = '<!-- metrics:start -->'
    end = '<!-- metrics:end -->'
    header = '| Spec | L | D | RVV bytes_read | bytes_written | mac_flops |\n|---|---|---|---|---|---|\n'
    body = ''.join([f"| {n} | {L} | {D} | {br} | {bw} | {mf} |\n" for (n,L,D,br,bw,mf) in rows])
    table = header + body
    if start in txt and end in txt:
        new = re.sub(re.escape(start)+'.*?'+re.escape(end), start+'\n'+table+'\n'+end, txt, flags=re.S)
    else:
        # append to Snapshot metrics section
        new = txt + '\n' + start + '\n' + table + '\n' + end + '\n'
    with open(p, 'w', encoding='utf-8') as f:
        f.write(new)


def main():
    rows = snapshot()
    update_readme(rows)
    print('Updated README metrics table with', len(rows), 'rows')


if __name__ == '__main__':
    main()


