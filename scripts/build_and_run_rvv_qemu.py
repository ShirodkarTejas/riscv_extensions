#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd, cwd=None):
    print('[run]', ' '.join(cmd))
    return subprocess.check_output(cmd, text=True, cwd=cwd)


def ensure_tools():
    for tool in ('riscv64-linux-gnu-gcc', 'qemu-riscv64'):
        if not shutil.which(tool):
            raise RuntimeError(f"Required tool '{tool}' not found on PATH")


def build(build_dir: str):
    os.makedirs(build_dir, exist_ok=True)
    tc = os.path.join('backends', 'rvv', 'toolchains', 'linux-gnu-rvv.cmake')
    run(['cmake', '-S', 'backends/rvv', '-B', build_dir, f'-DCMAKE_TOOLCHAIN_FILE={tc}', '-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake', '--build', build_dir, '-j'])


def qemu_prefix():
    prefix = []
    # Use target sysroot for dynamic linker and libs
    prefix += ['qemu-riscv64', '-L', '/usr/riscv64-linux-gnu']
    # Enable RVV and set a sane vector length
    prefix += ['-cpu', 'rv64,v=true,vlen=128,elen=64']
    return prefix


def main():
    ap = argparse.ArgumentParser(description='Cross-build RVV and run under QEMU user-mode')
    ap.add_argument('--build-dir', default='build/rvv-riscv64')
    ap.add_argument('--exe', default='sattn_rvv_compare_sw', help='Executable to run under QEMU')
    ap.add_argument('--args', default='', help='Additional args to pass to the executable')
    args = ap.parse_args()

    ensure_tools()
    build(args.build_dir)

    exe_path = os.path.join(args.build_dir, args.exe)
    if not os.path.exists(exe_path):
        raise FileNotFoundError(exe_path)

    cmd = qemu_prefix() + [exe_path] + ([a for a in args.args.split(' ') if a] if args.args else [])
    out = run(cmd)
    print(out)


if __name__ == '__main__':
    main()


