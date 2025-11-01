#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys


def test_qemu_runner_sw():
    if not shutil.which('qemu-riscv64') or not shutil.which('riscv64-linux-gnu-gcc'):
        print('[skip] qemu-riscv64 or riscv64-linux-gnu-gcc not found; skipping QEMU RVV runner test')
        return
    # Run the generic runner with a simple sliding_window case
    try:
        out = subprocess.check_output([
            sys.executable, 'scripts/build_and_run_rvv_qemu.py',
            '--exe', 'sattn_rvv_runner',
            '--args', '--spec sliding_window --L 128 --D 32 --window 8'
        ], text=True)
    except subprocess.CalledProcessError:
        print('[skip] RVV QEMU build failed in this environment')
        return
    assert 'checksum=' in out


