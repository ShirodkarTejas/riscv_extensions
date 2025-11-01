# sattn (Python)

Thin wrappers to run RVV kernels from MLIR and compile+simulate RoCC.

```python
from sattn import run_rvv_from_mlir, compile_and_sim

mlir = 'compiler/mlir/tests/_tmp_prof.mlir'
open(mlir,'w').write('module {\n  "sattn.sparse_attention"() { window_size = 8 : i64, tile_D = 16 : i64, tile_S = 64 : i64 } : () -> ()\n}\n')
print(run_rvv_from_mlir(mlir, prefer_bsr=True))
compile_and_sim(mlir, use_hw_probe=True)
```

Flags mirror the CLI tools (`--prefer-bsr`, `--prefer-sw`, `--l1-bytes`, `--use-hw-probe`, and `--autotune` for RVV).

