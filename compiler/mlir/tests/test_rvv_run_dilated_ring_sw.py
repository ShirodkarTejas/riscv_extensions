import subprocess, sys


def run_mlir(src):
    p = 'compiler/mlir/tests/_tmp_rvv_dilated.mlir'
    with open(p, 'w') as f:
        f.write(src)
    out = subprocess.check_output([sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p], text=True)
    return out


def test_dilated_sliding_window_runs():
    src = 'module {\n  "sattn.sparse_attention"() { spec = "sliding_window", window_size = 8 : i64, dilation = 2 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=sliding_window' in out


def test_ring_sliding_window_runs():
    src = 'module {\n  "sattn.sparse_attention"() { spec = "sliding_window", window_size = 8 : i64, wrap = 1 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    assert 'spec=sliding_window' in out


