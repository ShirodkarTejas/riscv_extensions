import os
import subprocess
import sys


def run_opt(src, env=None):
    p = 'compiler/mlir/tests/_tmp_select_hw.mlir'
    with open(p, 'w') as f:
        f.write(src)
    opt = 'build/mlir/tools/sattn-opt/sattn-opt'
    e = os.environ.copy()
    if env:
        e.update(env)
    out = subprocess.check_output([opt, p, '--allow-unregistered-dialect', '-pass-pipeline=builtin.module(sattn-lower-rvv)'], text=True, env=e)
    return out


def test_prefer_bsr_env():
    src = 'module {\n  "sattn.sparse_attention"() { block_size = 64 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_opt(src, env={'SATTN_PREFER_BSR': '1'})
    assert 'spec = "bsr"' in out or 'spec = "block_local_global"' in out


def test_prefer_sw_env():
    src = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_opt(src, env={'SATTN_PREFER_SW': '1'})
    assert 'spec = "sliding_window"' in out


