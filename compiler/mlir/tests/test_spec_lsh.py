#!/usr/bin/env python3
import os
import subprocess


def run(tool, src, pipeline):
    tmp = 'compiler/mlir/tests/_tmp_lsh.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect', f'-pass-pipeline=builtin.module({pipeline})']
    return subprocess.check_output(cmd, text=True)


def tool_path():
    return 'build/mlir/tools/sattn-opt/sattn-opt'


def test_lsh_selection_and_hook():
    tool = tool_path()
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { lsh_buckets = 64 : i64, tile_S = 512 : i64, tile_D = 64 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out
    assert 'spec = "lsh"' in out
    assert 'lsh_enabled = true' in out
    out2 = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out2
    assert 'spec = "lsh"' in out2
    assert 'lsh_enabled = true' in out2
