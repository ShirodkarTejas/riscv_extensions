import subprocess, sys


SRC = 'module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n'


def test_profile_rvv():
    p = 'compiler/mlir/tests/_tmp_prof.mlir'
    with open(p, 'w') as f:
        f.write(SRC)
    out = subprocess.check_output([sys.executable, 'scripts/sattn_profile.py', '--mlir', p, '--backend', 'rvv', '--prefer-bsr'], text=True)
    assert 'spec=' in out


def test_profile_sim():
    p = 'compiler/mlir/tests/_tmp_prof_sim.mlir'
    with open(p, 'w') as f:
        f.write(SRC)
    subprocess.check_call([sys.executable, 'scripts/sattn_profile.py', '--mlir', p, '--backend', 'sim', '--prefer-sw'])


