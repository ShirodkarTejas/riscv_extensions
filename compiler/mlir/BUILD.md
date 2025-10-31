# Building SATTN MLIR Components

This repository includes optional MLIR components (dialect/passes tooling). If MLIR is not found, the CMake files will skip building these targets.

## Prerequisites
- LLVM/MLIR build installed with `MLIRConfig.cmake` available on CMAKE_PREFIX_PATH

## Build
```bash
mkdir -p build && cd build
cmake .. -DSATTN_ENABLE_MLIR=ON -DCMAKE_PREFIX_PATH=/path/to/llvm-install
cmake --build . -j
```

If successful, you get `sattn-opt` that registers SATTN passes and wraps mlir-opt main.

```bash
./compiler/mlir/tools/sattn-opt/sattn-opt compiler/mlir/examples/sattn_example.mlir -o out.mlir
```

Note: The actual transformation logic will be implemented in the pass sources; initially, they are placeholders pending full integration.
