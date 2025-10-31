import os
import subprocess
import sys


def run_sim(mlir_src):
    p = 'compiler/mlir/tests/_tmp_gqa_comp.mlir'
    with open(p, 'w') as f:
        f.write(mlir_src)
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p]
    out = subprocess.check_output(cmd, text=True)
    return out


def parse_iter(s):
    for line in s.splitlines():
        if line.startswith('verilator_tb: completed in'):
            parts = line.split()
            return int(parts[3])
    raise AssertionError('iterations not found')


def test_gqa_increases_cycles():
    base = run_sim('module {\n  "sattn.sparse_attention"() { block_size = 4 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()\n}\n')
    gqa  = run_sim('module {\n  "sattn.sparse_attention"() { block_size = 4 : i64, gqa_group_size = 2 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()\n}\n')
    it_base = parse_iter(base)
    it_gqa = parse_iter(gqa)
    assert it_gqa >= it_base


def test_comp_blocks_reduce_cycles_when_smaller():
    base = run_sim('module {\n  "sattn.sparse_attention"() { block_size = 8 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()\n}\n')
    comp = run_sim('module {\n  "sattn.sparse_attention"() { block_size = 8 : i64, comp_block_size = 4 : i64, tile_D = 16 : i64, tile_S = 16 : i64 } : () -> ()\n}\n')
    it_base = parse_iter(base)
    it_comp = parse_iter(comp)
    assert it_comp <= it_base


