import importlib


def test_import_sattn_package():
    m = importlib.import_module('sattn')
    assert hasattr(m, 'run_rvv_from_mlir') and hasattr(m, 'compile_and_sim')


