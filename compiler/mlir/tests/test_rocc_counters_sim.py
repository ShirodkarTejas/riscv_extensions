#!/usr/bin/env python3
import subprocess
import sys


def test_rocc_counters_nonzero():
    src = 'compiler/mlir/tests/lower_to_rocc.mlir'
    out = subprocess.run([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', src], text=True, capture_output=True)
    assert out.returncode == 0
    s = out.stdout
    assert 'rocc_counters(rtl):' in s
    # Expect non-zero counters for our stub flows
    assert 'gather_cycles=0' not in s
    assert 'mac_cycles=0' not in s
    assert 'dma_bytes=0' not in s


