#!/usr/bin/env python3
import os
import shutil
import subprocess


def test_rocc_pipeline():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping rocc pipeline test')
        return
    # Minimal generic-ops MLIR (no func dialect needed)
    src = r'''
module {
  "sattn.sparse_attention"() { pattern = "block_topk", block_size = 16 : i64, keep_ratio = 0.25 : f32, precision = "bf16", softmax_mode = "logsumexp", tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 64 : i64 } : () -> ()
}
'''
    with open('compiler/mlir/tests/_tmp_rocc_input.mlir', 'w') as f:
        f.write(src)
    cmd = [tool, 'compiler/mlir/tests/_tmp_rocc_input.mlir', '--allow-unregistered-dialect',
           "-pass-pipeline=builtin.module(sattn-lower-rocc)"]
    print('[run]', ' '.join(cmd))
    out = subprocess.check_output(cmd, text=True)
    fc = shutil.which('FileCheck') or shutil.which('/usr/bin/FileCheck')
    if fc:
        # Create a small check file on the fly
        check = ' ; CHECK: sattn.rocc_call\n'
        p = subprocess.run([fc, '-'], input=out + check, text=True)
        assert p.returncode == 0, 'FileCheck failed'
    else:
        assert 'sattn.rocc_call' in out


