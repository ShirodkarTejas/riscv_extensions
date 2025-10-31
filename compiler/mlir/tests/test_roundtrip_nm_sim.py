#!/usr/bin/env python3
import subprocess
import sys


def test_roundtrip_nm_sim():
    mlir = 'compiler/mlir/tests/_tmp_nm.mlir'
    with open(mlir, 'w') as f:
        f.write('''module {
  "sattn.sparse_attention"() { nm_n = 2 : i64, nm_m = 4 : i64, tile_M = 16 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()
}
''')
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', mlir], text=True, capture_output=True)
    assert out.returncode == 0
    assert 'spec_info:' in out.stdout


