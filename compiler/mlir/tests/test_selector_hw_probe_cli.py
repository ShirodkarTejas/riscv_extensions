import subprocess, sys


def test_rvv_bridge_with_hw_probe_runs():
    src = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n'
    p = 'compiler/mlir/tests/_tmp_hw_probe_rvv.mlir'
    with open(p, 'w') as f:
        f.write(src)
    out = subprocess.check_output([sys.executable, 'compiler/mlir/tools/sattn_run_rvv_from_mlir.py', '--mlir', p, '--use-hw-probe'], text=True)
    assert 'spec=' in out


def test_sim_wrapper_with_hw_probe_runs():
    src = 'module {\n  "sattn.sparse_attention"() { block_size = 4 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n'
    p = 'compiler/mlir/tests/_tmp_hw_probe_sim.mlir'
    with open(p, 'w') as f:
        f.write(src)
    # Should run sim probe and full flow
    subprocess.check_call([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p, '--use-hw-probe'])


