#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys


def test_qemu_compare_sw():
    if not shutil.which('qemu-riscv64') or not shutil.which('riscv64-linux-gnu-gcc'):
        print('[skip] qemu-riscv64 or riscv64-linux-gnu-gcc not found; skipping QEMU RVV test')
        return
    out = subprocess.check_output([sys.executable, 'scripts/build_and_run_rvv_qemu.py'], text=True)
    assert 'MATCH' in out



