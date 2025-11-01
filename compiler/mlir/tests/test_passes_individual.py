#!/usr/bin/env python3
import os
import shutil
import subprocess


def has_tool():
    return os.path.exists('build/mlir/tools/sattn-opt/sattn-opt')


def test_tile_only_sets_defaults():
    if not has_tool():
        print('[skip] sattn-opt not found; skipping individual pass test (tile)')
        return
    src = 'module { "sattn.sparse_attention"() : () -> () }\n'
    p = 'builtin.module(sattn-tile)'
    out = subprocess.check_output(['build/mlir/tools/sattn-opt/sattn-opt', '--allow-unregistered-dialect', '-pass-pipeline=' + p], input=src, text=True)
    assert 'tile_M = 64' in out and 'tile_D = 64' in out and 'tile_S = 128' in out


def test_lower_rvv_then_bufferize():
    if not has_tool():
        print('[skip] sattn-opt not found; skipping individual pass test (bufferize)')
        return
    src = 'module { "sattn.sparse_attention"() { tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> () }\n'
    p = 'builtin.module(sattn-lower-to-rvv,sattn-bufferize)'
    out = subprocess.check_output(['build/mlir/tools/sattn-opt/sattn-opt', '--allow-unregistered-dialect', '-pass-pipeline=' + p], input=src, text=True)
    assert 'sattn.rvv_call' in out and 'bufferized' in out and 'buffer_layout = "rowmajor"' in out


