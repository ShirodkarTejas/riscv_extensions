#!/usr/bin/env python3
import os
import subprocess


def run(tool, src, pipeline):
    tmp = 'compiler/mlir/tests/_tmp_attrs.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect', f'-pass-pipeline=builtin.module({pipeline})']
    return subprocess.check_output(cmd, text=True)


def tool_path():
    return 'build/mlir/tools/sattn-opt/sattn-opt'


def test_nm_attrs_propagate():
    tool = tool_path()
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { nm_n = 2 : i64, nm_m = 4 : i64, tile_S = 128 : i64, tile_D = 32 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out
    assert 'nm_enabled = true' in out
    assert 'nm_n = 2' in out and 'nm_m = 4' in out
    out2 = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out2
    assert 'nm_enabled = true' in out2
    assert 'nm_n = 2' in out2 and 'nm_m = 4' in out2


def test_lsh_attrs_propagate():
    tool = tool_path()
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { lsh_buckets = 8 : i64, tile_S = 256 : i64, tile_D = 64 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out
    assert 'lsh_enabled = true' in out
    assert 'lsh_buckets = 8' in out
    out2 = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out2
    assert 'lsh_enabled = true' in out2
    assert 'lsh_buckets = 8' in out2
