import os
import subprocess
import sys
from typing import Optional


def run_rvv_from_mlir(mlir: str, prefer_bsr: bool=False, prefer_sw: bool=False, l1_bytes: int=0, use_hw_probe: bool=False, autotune: bool=False) -> str:
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', mlir]
    if prefer_bsr: cmd += ['--prefer-bsr']
    if prefer_sw: cmd += ['--prefer-sw']
    if l1_bytes: cmd += ['--l1-bytes', str(l1_bytes)]
    if use_hw_probe: cmd += ['--use-hw-probe']
    if autotune: cmd += ['--autotune']
    return subprocess.check_output(cmd, text=True)


def compile_and_sim(mlir: str, prefer_bsr: bool=False, prefer_sw: bool=False, l1_bytes: int=0, use_hw_probe: bool=False) -> None:
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', mlir]
    if prefer_bsr: cmd += ['--prefer-bsr']
    if prefer_sw: cmd += ['--prefer-sw']
    if l1_bytes: cmd += ['--l1-bytes', str(l1_bytes)]
    if use_hw_probe: cmd += ['--use-hw-probe']
    subprocess.check_call(cmd)


