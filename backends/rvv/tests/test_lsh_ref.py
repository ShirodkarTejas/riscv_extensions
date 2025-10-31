#!/usr/bin/env python3
import os
import subprocess


def test_compare_lsh_ref():
    exe = 'build/rvv/sattn_rvv_compare_lsh'
    if not os.path.exists(exe):
        assert False, f"{exe} not built"
    out = subprocess.check_output([exe], text=True)
    assert 'MATCH' in out


