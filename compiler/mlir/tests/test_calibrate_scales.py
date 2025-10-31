import subprocess, sys


def run_calib(src, precision):
    p = 'compiler/mlir/tests/_tmp_calib.mlir'
    with open(p, 'w') as f:
        f.write(src)
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_calibrate_scales.py', '--mlir', p, '--precision', precision]
    out = subprocess.check_output(cmd, text=True)
    return out


def test_calibrate_i8():
    src = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_calib(src, 'i8')
    assert 'calibrate:' in out and 'scale_q_x1000=' in out


def test_calibrate_i4():
    src = 'module {\n  "sattn.sparse_attention"() { global_tokens = 4 : i64, block_size = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_calib(src, 'i4')
    assert 'calibrate:' in out and 'scale_v_x1000=' in out


