#!/usr/bin/env python3
import subprocess
import sys
import re


def run_mlir(mlir_src):
    p = 'compiler/mlir/tests/_tmp_dma.mlir'
    with open(p, 'w') as f: f.write(mlir_src)
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True, capture_output=True)
    return out.stdout


def parse_dma(s):
    m = re.search(r'dma_bytes=(\d+)', s)
    assert m, 'dma_bytes not found'
    return int(m.group(1))


def test_dma_bytes_nonzero_and_aligned():
    S, D = 64, 32
    out = run_mlir(f'module {{\n  "sattn.sparse_attention"() {{ tile_M = 16 : i64, tile_D = {D} : i64, tile_S = {S} : i64 }} : () -> ()\n}}\n')
    dma = parse_dma(out)
    assert dma > 0 and (dma % 8) == 0


