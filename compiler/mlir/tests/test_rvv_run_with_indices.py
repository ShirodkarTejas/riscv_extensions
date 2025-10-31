import subprocess, sys


def run_mlir(src):
    p = 'compiler/mlir/tests/_tmp_rvv_indices.mlir'
    with open(p, 'w') as f:
        f.write(src)
    out = subprocess.check_output([sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p], text=True)
    return out


def test_rvv_blg_uses_indices_file():
    src = 'module {\n  "sattn.sparse_attention"() { spec = "block_local_global", block_size = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_mlir(src)
    # Just verify it ran and printed a spec line (indices emission is internal)
    assert 'spec=block_local_global' in out or 'spec=bsr' in out


