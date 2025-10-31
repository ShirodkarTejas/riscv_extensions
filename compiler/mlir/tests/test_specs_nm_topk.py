#!/usr/bin/env python3
import os
import subprocess


def run(tool, src, pipeline):
    tmp = 'compiler/mlir/tests/_tmp_specs.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect', f'-pass-pipeline=builtin.module({pipeline})']
    return subprocess.check_output(cmd, text=True)


def tool_path():
    return 'build/mlir/tools/sattn-opt/sattn-opt'


def test_nm_structured_selection_and_hook():
    tool = tool_path()
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { nm_n = 2 : i64, nm_m = 4 : i64, tile_S = 128 : i64, tile_D = 32 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out
    assert 'spec = "nm_structured"' in out
    assert 'nm_enabled = true' in out
    out2 = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out2
    assert 'spec = "nm_structured"' in out2
    assert 'nm_enabled = true' in out2


def test_topk_selection_and_hook():
    tool = tool_path()
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { topk_k = 8 : i64, tile_S = 128 : i64, tile_D = 32 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rvv')
    assert 'sattn.rvv_call' in out
    assert 'spec = "topk_per_query"' in out
    assert 'topk_enabled = true' in out
    out2 = run(tool, src, 'sattn-lower-rocc')
    assert 'sattn.rocc_call' in out2
    assert 'spec = "topk_per_query"' in out2
    assert 'topk_enabled = true' in out2
