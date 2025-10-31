import os
import shutil
import subprocess
import sys


def have_filecheck():
    return shutil.which('FileCheck') is not None


def run_opt(src, pipeline='builtin.module(sattn-lower-rvv)'):
    p = 'compiler/mlir/tests/_tmp_select_fc.mlir'
    with open(p, 'w') as f:
        f.write(src)
    opt = 'build/mlir/tools/sattn-opt/sattn-opt'
    if not os.path.exists(opt):
        raise pytest.skip('sattn-opt not available')
    out = subprocess.check_output([opt, p, '--allow-unregistered-dialect', f'-pass-pipeline={pipeline}'], text=True)
    return out


def test_select_prefers_bsr_with_gqa():
    # With gqa_group_size > 1 and no window_size, selector should pick bsr/BLG
    src = 'module {\n  "sattn.sparse_attention"() { gqa_group_size = 2 : i64, block_size = 64 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_opt(src)
    if have_filecheck():
        fc = subprocess.Popen(['FileCheck', '-check-prefix=GQA'], stdin=subprocess.PIPE, text=True)
        fc.stdin.write(out)
        fc.stdin.close()
        fc.wait()
        assert fc.returncode == 0
    else:
        assert 'spec = "bsr"' in out or 'spec = "block_local_global"' in out


def test_select_discounts_bsr_with_comp():
    # With comp_block_size < block_size, block-based selection gets discounted
    src = 'module {\n  "sattn.sparse_attention"() { comp_block_size = 32 : i64, block_size = 64 : i64, tile_D = 32 : i64, tile_S = 128 : i64 } : () -> ()\n}\n'
    out = run_opt(src)
    if have_filecheck():
        fc = subprocess.Popen(['FileCheck', '-check-prefix=COMP'], stdin=subprocess.PIPE, text=True)
        fc.stdin.write(out)
        fc.stdin.close()
        fc.wait()
        assert fc.returncode == 0
    else:
        assert 'spec = "bsr"' in out or 'spec = "block_local_global"' in out


