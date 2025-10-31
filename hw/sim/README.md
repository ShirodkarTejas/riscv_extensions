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
```

Expected output:
```
verilator_tb: completed in <N> iterations
```

This harness programs a minimal descriptor and issues `spdot_bsr` (command 0x14), then polls the status register until the DONE bit is set by the fixed-latency FSM.
