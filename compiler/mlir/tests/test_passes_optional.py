#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import re


def main():
    candidates = [
        'build/mlir/tools/sattn-opt/sattn-opt',
        'compiler/mlir/tools/sattn-opt/sattn-opt',
    ]
    tool = next((p for p in candidates if os.path.exists(p)), None)
    if not tool:
        print('[skip] sattn-opt not found; skipping pass checks')
        return 0
    # Minimal smoke test: tool should execute and print help
    cmd = [tool, '--help']
    print('[run]', ' '.join(cmd))
    subprocess.check_call(cmd)
    print('[pass] sattn-opt is built and runnable')
    return 0


if __name__ == '__main__':
    sys.exit(main())


