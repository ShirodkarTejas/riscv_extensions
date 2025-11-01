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
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    gcc_tc = os.path.abspath(os.path.join('backends', 'rvv', 'toolchains', 'linux-gnu-rvv.cmake'))
    clang_tc = os.path.abspath(os.path.join('backends', 'rvv', 'toolchains', 'linux-gnu-rvv-clang.cmake'))
    try:
        run(['cmake', '-G', 'Ninja', '-S', 'backends/rvv', '-B', build_dir,
             f'-DCMAKE_TOOLCHAIN_FILE={gcc_tc}', '-DCMAKE_BUILD_TYPE=Release'])
        run(['cmake', '--build', build_dir, '-j'])
    except subprocess.CalledProcessError:
        # Retry with Clang toolchain
        shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)
        # Patch absolute paths in glibc linker script to be sysroot-friendly
        libc_script = '/usr/riscv64-linux-gnu/lib/libc.so'
        try:
            with open(libc_script, 'r', encoding='utf-8') as f:
                txt = f.read()
            new_txt = txt.replace('/usr/riscv64-linux-gnu/lib/', '')
            if new_txt != txt:
                with open(libc_script, 'w', encoding='utf-8') as f:
                    f.write(new_txt)
        except Exception as _:
            pass
        run(['cmake', '-G', 'Ninja', '-S', 'backends/rvv', '-B', build_dir,
             f'-DCMAKE_TOOLCHAIN_FILE={clang_tc}', '-DCMAKE_BUILD_TYPE=Release'])
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


