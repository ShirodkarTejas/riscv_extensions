#!/usr/bin/env python3
import os
import subprocess


def run(tool, src, pipeline):
    tmp = 'compiler/mlir/tests/_tmp_blg.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect', f'-pass-pipeline=builtin.module({pipeline})']
    return subprocess.check_output(cmd, text=True)


def test_blg_rocc_selected():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { global_tokens = 8 : i64, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out
    assert 'spec = "block_local_global"' in out
    assert 'blg_enabled = true' in out


def test_blg_rvv_selected():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { global_tokens = 4 : i64, block_size = 64 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out
    assert 'spec = "block_local_global"' in out
    assert 'blg_enabled = true' in out
