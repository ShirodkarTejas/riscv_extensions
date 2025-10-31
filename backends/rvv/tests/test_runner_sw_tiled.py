#!/usr/bin/env python3
import os
import subprocess


def test_runner_sw_vs_tiled_checksum_match():
    exe = 'build/rvv/sattn_rvv_runner'
    if not os.path.exists(exe):
        assert False, f"{exe} not built"
    base = subprocess.check_output([exe, '--spec', 'sliding_window', '--L', '128', '--D', '32', '--window', '8'], text=True)
    tiled = subprocess.check_output([exe, '--spec', 'sliding_window', '--L', '128', '--D', '32', '--window', '8', '--tile_rows', '4'], text=True)
    assert 'spec=sliding_window' in base and 'checksum=' in base
    assert 'spec=sliding_window' in tiled and 'checksum=' in tiled
    c0 = float(base.strip().split('checksum=')[-1].split()[0])
    c1 = float(tiled.strip().split('checksum=')[-1].split()[0])
    assert abs(c0 - c1) < 1e-4


