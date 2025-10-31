#!/usr/bin/env python3
import os
import subprocess


def run(tool, src):
    tmp = 'compiler/mlir/tests/_tmp_select_spec.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect',
           "-pass-pipeline=builtin.module(sattn-lower-rocc)"]
    return subprocess.check_output(cmd, text=True)


def test_select_sliding_window():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { window_size = 15 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()
}
'''
    out = run(tool, src)
    assert 'sattn.rocc_call' in out
    assert 'spec = "sliding_window"' in out


def test_select_bsr_default():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()
}
'''
    out = run(tool, src)
    assert 'sattn.rocc_call' in out
    assert 'spec = "bsr"' in out


