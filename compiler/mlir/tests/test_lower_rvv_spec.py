#!/usr/bin/env python3
import os
import subprocess


def test_lower_rvv_propagates_spec():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping RVV test')
        return
    src = r'''module {
  "sattn.sparse_attention"() { window_size = 32 : i64, keep_ratio = 0.9 : f32, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 256 : i64 } : () -> ()
}
'''
    tmp = 'compiler/mlir/tests/_tmp_rvv_input.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect',
           '-pass-pipeline=builtin.module(sattn-lower-rvv)']
    print('[run]', ' '.join(cmd))
    out = subprocess.check_output(cmd, text=True)
    assert 'sattn.rvv_call' in out
    assert 'spec = "sliding_window"' in out


