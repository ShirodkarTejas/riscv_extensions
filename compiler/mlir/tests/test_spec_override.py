#!/usr/bin/env python3
import os
import subprocess


def run(tool, src, pipeline, env=None):
    tmp = 'compiler/mlir/tests/_tmp_override.mlir'
    with open(tmp, 'w') as f:
        f.write(src)
    cmd = [tool, tmp, '--allow-unregistered-dialect', f'-pass-pipeline=builtin.module({pipeline})']
    env2 = os.environ.copy()
    if env:
        env2.update(env)
    return subprocess.check_output(cmd, text=True, env=env2)


def test_env_force_spec_bsr():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    # Would normally pick sliding_window, but env override forces bsr
    src = r'''module {
  "sattn.sparse_attention"() { window_size = 8 : i64, tile_S = 128 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rvv', env={'SATTN_FORCE_SPEC': 'bsr'})
    assert 'sattn.rvv_call' in out
    assert 'spec = "bsr"' in out


def test_disable_sw_prefers_bsr():
    tool = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(tool):
        print('[skip] sattn-opt not found; skipping')
        return
    src = r'''module {
  "sattn.sparse_attention"() { window_size = 8 : i64, tile_S = 128 : i64 } : () -> ()
}
'''
    out = run(tool, src, 'sattn-lower-rocc', env={'SATTN_DISABLE_SW': '1'})
    assert 'sattn.rocc_call' in out
    assert 'spec = "bsr"' in out
