#!/usr/bin/env python3
import subprocess
import sys
import re


def run_mlir(src):
    p = 'compiler/mlir/tests/_tmp_dma_split.mlir'
    with open(p, 'w') as f: f.write(src)
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True, capture_output=True)
    return out.stdout


def test_dma_split_sum_matches_total():
    s = run_mlir('module {\n  "sattn.sparse_attention"() { tile_M = 8 : i64, tile_D = 16 : i64, tile_S = 32 : i64 } : () -> ()\n}\n')
    m = re.search(r'dma_bytes=(\d+) dma_q=(\d+) dma_k=(\d+)', s)
    assert m, 'split counters not found'
    total, q, k = map(int, m.groups())
    assert total == q + k and total > 0 and q > 0 and k > 0
    assert q == k


