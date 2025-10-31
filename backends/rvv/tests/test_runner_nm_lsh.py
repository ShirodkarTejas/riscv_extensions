#!/usr/bin/env python3
import os
import subprocess


def test_runner_nm_structured():
    exe = 'build/rvv/sattn_rvv_runner'
    if not os.path.exists(exe):
        assert False, f"{exe} not built"
    out = subprocess.check_output([exe, '--spec', 'nm_structured', '--L', '128', '--D', '32', '--nm_n', '2', '--nm_m', '4'], text=True)
    assert 'spec=nm_structured' in out and 'checksum=' in out


def test_runner_lsh():
    exe = 'build/rvv/sattn_rvv_runner'
    if not os.path.exists(exe):
        assert False, f"{exe} not built"
    out = subprocess.check_output([exe, '--spec', 'lsh', '--L', '128', '--D', '32', '--lsh_buckets', '8'], text=True)
    assert 'spec=lsh' in out and 'checksum=' in out


