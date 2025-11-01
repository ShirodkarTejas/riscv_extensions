import subprocess, sys


def run_sim(tile_S, tile_D):
    src = (
        'module {\n'
        f'  "sattn.sparse_attention"() {{ block_size = 4 : i64, tile_D = {tile_D} : i64, tile_S = {tile_S} : i64 }} : () -> ()\n'
        '}\n'
    )
    p = 'compiler/mlir/tests/_tmp_rocc_invar.mlir'
    with open(p, 'w') as f:
        f.write(src)
    out = subprocess.check_output([sys.executable, 'compiler/mlir/tools/sattn_compile_and_sim.py', '--mlir', p], text=True)
    return out


def parse_counters(out):
    g = m = b = bq = bk = None
    ug = um = None
    for line in out.splitlines():
        if line.startswith('rocc_counters(rtl):'):
            # rocc_counters(rtl):   gather_cycles=X mac_cycles=Y dma_bytes=Z dma_q=W dma_k=K
            parts = line.split()
            for p in parts:
                if p.startswith('gather_cycles='): g = int(p.split('=')[1])
                elif p.startswith('mac_cycles='): m = int(p.split('=')[1])
                elif p.startswith('dma_bytes='): b = int(p.split('=')[1])
                elif p.startswith('dma_q='): bq = int(p.split('=')[1])
                elif p.startswith('dma_k='): bk = int(p.split('=')[1])
        if line.startswith('rocc_util:'):
            parts = line.replace(',', ' ').split()
            for p in parts:
                if p.startswith('util_mac='): um = float(p.split('=')[1])
                elif p.startswith('util_gather='): ug = float(p.split('=')[1])
    return g, m, b, bq, bk, um, ug


def test_rocc_invariants_monotonic_and_util():
    out1 = run_sim(64, 16)
    g1, m1, b1, bq1, bk1, um1, ug1 = parse_counters(out1)
    out2 = run_sim(128, 16)
    g2, m2, b2, bq2, bk2, um2, ug2 = parse_counters(out2)
    # bytes split exact
    assert b1 == bq1 + bk1
    assert b2 == bq2 + bk2
    # monotonic in S
    assert g2 >= g1 and m2 >= m1 and b2 >= b1
    # utilization within bounds
    for u in (um1, ug1, um2, ug2):
        assert 0.0 <= u <= 1.05


