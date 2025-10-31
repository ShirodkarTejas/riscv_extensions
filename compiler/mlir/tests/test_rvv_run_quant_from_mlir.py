import subprocess, sys


def run_mlir(src):
    p = 'compiler/mlir/tests/_tmp_rvv_quant.mlir'
    with open(p, 'w') as f:
        f.write(src)
    cmd = [sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p]
    out = subprocess.check_output(cmd, text=True)
    return out


def parse_checksum(s):
    for line in s.splitlines():
        if line.startswith('spec=') and 'checksum=' in line:
            for tok in line.split():
                if tok.startswith('checksum='):
                    return float(tok.split('=')[1])
    raise AssertionError('checksum not found')


def test_quant_bf16_from_mlir():
    # Include precision attribute; pipeline will lower and preserve it
    src = 'module {\n  "sattn.sparse_attention"() { precision = "bf16", window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out_bf16 = run_mlir(src)
    # Baseline fp32
    src2 = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out_fp32 = run_mlir(src2)
    c_bf16 = parse_checksum(out_bf16)
    c_fp32 = parse_checksum(out_fp32)
    assert abs(c_fp32 - c_bf16) / (abs(c_fp32) + 1e-9) < 1e-2


def test_quant_i8_from_mlir():
    src = 'module {\n  "sattn.sparse_attention"() { precision = "i8", scale_q = 0.05 : f32, scale_k = 0.05 : f32, scale_v = 0.05 : f32, window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out_i8 = run_mlir(src)
    assert ('spec=sliding_window' in out_i8) or ('spec=block_local_global' in out_i8) or ('spec=bsr' in out_i8)


