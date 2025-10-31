#!/usr/bin/env python3
import subprocess
import sys
import re


def run_mlir(mlir_src):
    p = 'compiler/mlir/tests/_tmp_util.mlir'
    with open(p, 'w') as f: f.write(mlir_src)
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True, capture_output=True)
    return out.stdout


def test_utilization_bounds():
    s = run_mlir('module {\n  "sattn.sparse_attention"() { tile_M = 8 : i64, tile_D = 16 : i64, tile_S = 32 : i64 } : () -> ()\n}\n')
    m = re.search(r'rocc_util: util_mac=([0-9]+\.[0-9]+) util_gather=([0-9]+\.[0-9]+)', s)
    assert m, 'util line not found'
    util_mac = float(m.group(1)); util_gather = float(m.group(2))
    assert 0.0 <= util_mac <= 1.05
    assert util_gather >= 0.0


