#!/usr/bin/env python3
import subprocess
import sys


def main():
    mlir = 'compiler/mlir/tests/lower_to_rocc.mlir'
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', mlir]
    print('[test]', ' '.join(cmd))
    subprocess.check_call(cmd)
    print('[test] PASS')


if __name__ == '__main__':
    main()


