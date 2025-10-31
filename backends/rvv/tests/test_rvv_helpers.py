#!/usr/bin/env python3
import os
import subprocess


def test_rvv_helpers_pass():
    exe = 'build/rvv/sattn_rvv_test_helpers'
    if not os.path.exists(exe):
        assert False, f"{exe} not built"
    out = subprocess.check_output([exe], text=True)
    assert 'RVV helpers PASS' in out


