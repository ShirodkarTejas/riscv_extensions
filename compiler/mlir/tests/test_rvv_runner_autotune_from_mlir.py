import subprocess, sys


def run_mlir(src, autotune=False):
    p = 'compiler/mlir/tests/_tmp_rvv_autotune.mlir'
    with open(p, 'w') as f:
        f.write(src)
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p]
    if autotune:
        cmd.append('--autotune')
    out = subprocess.check_output(cmd, text=True)
    return out


def test_autotune_flag_runs():
    src = 'module {\n  "sattn.sparse_attention"() { spec = "sliding_window", window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src, autotune=True)
    # Expect autotune line from runner
    assert 'autotune:' in out


