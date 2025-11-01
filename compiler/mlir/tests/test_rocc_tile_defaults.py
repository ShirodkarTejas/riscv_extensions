#!/usr/bin/env python3
import os
import shutil
import subprocess


def test_rocc_tile_defaults_propagate():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping tile defaults test')
        return
    src = r'''
module {
  "sattn.sparse_attention"() : () -> ()
}
'''
    with open('compiler/mlir/tests/_tmp_rocc_tile_input.mlir', 'w') as f:
        f.write(src)
    cmd = [tool, 'compiler/mlir/tests/_tmp_rocc_tile_input.mlir', '--allow-unregistered-dialect',
           "-pass-pipeline=builtin.module(sattn-lower-rocc)"]
    out = subprocess.check_output(cmd, text=True)
    fc = shutil.which('FileCheck') or shutil.which('/usr/bin/FileCheck')
    if fc:
        # Expect the rocc_call to have m_rows=64, head_dim_d=64, s_tokens=128 (Tile defaults)
        check = (
            ' ; CHECK: sattn.rocc_call\n'
            ' ; CHECK: m_rows = 64\n'
            ' ; CHECK: head_dim_d = 64\n'
            ' ; CHECK: s_tokens = 128\n'
        )
        p = subprocess.run([fc, '-'], input=out + check, text=True)
        assert p.returncode == 0, 'FileCheck failed'
    else:
        assert 'sattn.rocc_call' in out
        assert 'm_rows = 64' in out
        assert 'head_dim_d = 64' in out
        assert 's_tokens = 128' in out


