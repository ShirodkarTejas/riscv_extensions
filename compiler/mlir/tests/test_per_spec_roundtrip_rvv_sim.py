import subprocess, sys


def run(cmd):
    return subprocess.check_output(cmd, text=True)


def run_rvv_from_mlir(src):
    p = 'compiler/mlir/tests/_tmp_per_spec_rvv.mlir'
    with open(p, 'w') as f:
        f.write(src)
    return run([sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p])


def run_sim_from_mlir(src):
    p = 'compiler/mlir/tests/_tmp_per_spec_sim.mlir'
    with open(p, 'w') as f:
        f.write(src)
    return run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p])


CASES = [
    'module {\n  "sattn.sparse_attention"() { spec = "sliding_window", window_size = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n',
    'module {\n  "sattn.sparse_attention"() { spec = "block_local_global", block_size = 16 : i64, keep_ratio = 0.12 : f32, global_tokens = 4 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n',
    'module {\n  "sattn.sparse_attention"() { spec = "nm_structured", nm_n = 2 : i64, nm_m = 4 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n',
    'module {\n  "sattn.sparse_attention"() { spec = "lsh", lsh_buckets = 8 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n',
    'module {\n  "sattn.sparse_attention"() { spec = "landmark", num_landmarks = 16 : i64, landmark_iters = 1 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n',
]


def test_per_spec_rvv_and_sim_roundtrip():
    for src in CASES:
        out_rvv = run_rvv_from_mlir(src)
        assert 'spec=' in out_rvv
        out_sim = run_sim_from_mlir(src)
        # Sim prints these headers on success
        assert 'verilator_tb: completed' in out_sim and 'rocc_counters(rtl):' in out_sim


