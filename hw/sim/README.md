# RoCC Verilator Simulation Harness

Build and run the skeleton accelerator in simulation.

Requirements:
- Verilator installed and on PATH

Build:
```bash
cd hw/sim
make
```

Run:
```bash
./obj_dir/Vrocc_sattn
# or pass an indices file (one decimal per line) to prefill index RAM
./obj_dir/Vrocc_sattn indices.txt
```

Expected output:
```
verilator_tb: completed in <N> iterations
```

This harness programs a minimal descriptor and issues `spdot_bsr` (command 0x14), then polls the status register until the DONE bit is set by the fixed-latency FSM.
If an indices file is provided, it writes those entries into the on-chip index RAM before issuing the command.

Generate indices from MLIR:
```bash
python compiler/mlir/tools/sattn_emit_indices_txt.py \
  --in-mlir compiler/mlir/examples/sattn_example.mlir \
  --out-indices indices.txt
./obj_dir/Vrocc_sattn indices.txt

# Or run the helper script that emits indices and runs the sim
python hw/sim/run_from_mlir.py --mlir compiler/mlir/examples/sattn_example.mlir
```
