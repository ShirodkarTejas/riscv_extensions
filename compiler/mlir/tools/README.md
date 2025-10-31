# Tools and CLI

Planned CLI integration for the SATTN dialect and passes.

- `sattn-opt` (future): a thin wrapper around `mlir-opt` with our passes registered.
- Example end-to-end:
```
mlir-opt input.mlir \
  -sattn-materialize-indices -sattn-tile -sattn-fuse-softmax \
  -sattn-lower-to-rocc -convert-vector-to-llvm | \
mlir-translate --mlir-to-llvmir > out.ll
```

This repository currently ships the TableGen op definition and pass documentation so you can start modeling schedules and attributes before wiring a full build.
