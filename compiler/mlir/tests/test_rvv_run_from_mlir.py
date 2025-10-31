#!/usr/bin/env python3
import os
import subprocess
import sys


def run_mlir(src):
    p = 'compiler/mlir/tests/_tmp_rvv_run.mlir'
    with open(p, 'w') as f: f.write(src)
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p]
    print('[test]', ' '.join(cmd))
    out = subprocess.check_output(cmd, text=True)
    return out


def test_run_sw():
    # Bias selection toward sliding_window by making BSR keep_ratio high
    src = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, keep_ratio = 0.5 : f32, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=sliding_window' in out


def test_run_blg():
    src = 'module {\n  "sattn.sparse_attention"() { global_tokens = 8 : i64, block_size = 64 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=block_local_global' in out or 'spec=bsr' in out


def test_run_nm():
    src = 'module {\n  "sattn.sparse_attention"() { nm_n = 2 : i64, nm_m = 4 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=nm_structured' in out


def test_run_lsh():
    src = 'module {\n  "sattn.sparse_attention"() { lsh_buckets = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=lsh' in out


