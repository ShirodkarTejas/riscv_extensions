import subprocess, sys


def run_sim(src):
    p = 'compiler/mlir/tests/_tmp_rocc_cpu_mlir.mlir'
    with open(p, 'w') as f:
        f.write(src)
    out = subprocess.check_output([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True)
    return out


CASES = [
    'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n',
    'module {\n  "sattn.sparse_attention"() { block_size = 4 : i64, keep_ratio = 0.12 : f32, global_tokens = 2 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n',
]


def test_rocc_checksum_matches_cpu_reference():
    for src in CASES:
        out = run_sim(src)
        # Testbench prints expected=... -> PASS when checksum matches reference
        assert '-> PASS' in out


