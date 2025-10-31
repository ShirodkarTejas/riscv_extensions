#!/usr/bin/env python3
import os
import subprocess
import sys


def run(cmd, cwd=None):
    print('[run]', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    # Assume running inside dev container at /workspace
    # Build MLIR tool and RVV backends
    run(['/usr/bin/cmake', '--build', 'build/mlir', '-j'])
    run(['/usr/bin/cmake', '--build', 'build/rvv', '-j'])

    # Run MLIR tests
    run(['/opt/venv/bin/python', '-m', 'pytest', '-q', 'compiler/mlir/tests'])
    # Run RVV tests
    run(['/opt/venv/bin/python', '-m', 'pytest', '-q', 'backends/rvv/tests'])

    print('All tests PASS')


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()


