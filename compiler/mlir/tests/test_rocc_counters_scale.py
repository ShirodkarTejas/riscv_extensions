#!/usr/bin/env python3
import subprocess
import sys
import re


def run_mlir_and_capture(mlir_src):
    p = 'compiler/mlir/tests/_tmp_scale.mlir'
    with open(p, 'w') as f: f.write(mlir_src)
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True, capture_output=True)
    return out.stdout


def parse_counters(s):
    m = re.search(r'rocc_counters\(rtl\):\s+gather_cycles=(\d+) mac_cycles=(\d+) dma_bytes=(\d+)', s)
    assert m, 'rtl counters not found'
    return tuple(map(int, m.groups()))


def test_counters_scale_with_S():
    base = run_mlir_and_capture('module {\n  "sattn.sparse_attention"() { tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()\n}\n')
    big  = run_mlir_and_capture('module {\n  "sattn.sparse_attention"() { tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n')
    g0, m0, d0 = parse_counters(base)
    g1, m1, d1 = parse_counters(big)
    assert g1 >= g0 and m1 >= m0 and d1 >= d0


