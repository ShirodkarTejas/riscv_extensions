#!/usr/bin/env python3
import os
import shutil
import subprocess


def test_rvv_pipeline():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping rvv pipeline test')
        return
    src = r'''
module {
  "sattn.sparse_attention"() { window_size = 8 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
'''
    with open('compiler/mlir/tests/_tmp_rvv_input.mlir', 'w') as f:
        f.write(src)
    cmd = [tool, 'compiler/mlir/tests/_tmp_rvv_input.mlir', '--allow-unregistered-dialect',
           "-pass-pipeline=builtin.module(sattn-lower-rvv)"]
    print('[run]', ' '.join(cmd))
    out = subprocess.check_output(cmd, text=True)
    fc = shutil.which('FileCheck') or shutil.which('/usr/bin/FileCheck')
    if fc:
        check = ' ; CHECK: sattn.rvv_call\n'
        p = subprocess.run([fc, '-'], input=out + check, text=True)
        assert p.returncode == 0, 'FileCheck failed'
    else:
        assert 'sattn.rvv_call' in out


