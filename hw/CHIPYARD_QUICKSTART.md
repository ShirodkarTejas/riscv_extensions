# Chipyard Integration Quick Start

This guide explains how to integrate the sparse attention accelerator with Chipyard and run hardware simulations.

---

## Prerequisites

- Linux machine (Ubuntu 20.04+ recommended)
- 16+ GB RAM
- 50+ GB disk space
- Java 8 or 11
- Python 3.7+

---

## Step 1: Install Chipyard

```bash
# Clone Chipyard
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard

# Initialize submodules
./build-setup.sh riscv-tools

# This will take 30-60 minutes
```

---

## Step 2: Copy Sparse Attention Accelerator

```bash
# From your riscv_extensions directory
cd /path/to/riscv_extensions

# Copy Chisel code to Chipyard
mkdir -p /path/to/chipyard/generators/sattn
cp hw/chisel/* /path/to/chipyard/generators/sattn/

# Copy Verilog modules
cp hw/rtl/*.sv /path/to/chipyard/generators/sattn/
```

---

## Step 3: Update Chipyard Build Configuration

Edit `chipyard/build.sbt`:

```scala
lazy val sattn = (project in file("generators/sattn"))
  .dependsOn(rocketchip)
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "edu.berkeley.cs" %% "chisel3" % "3.6.0",
      "edu.berkeley.cs" %% "rocketchip" % "1.6.0"
    )
  )
```

Update root project dependencies:

```scala
lazy val chipyard = (project in file("."))
  .dependsOn(sattn) // Add this line
  .dependsOn(...)
```

---

## Step 4: Add Configuration to Chipyard

Create `chipyard/src/main/scala/config/SattnConfigs.scala`:

```scala
package chipyard

import org.chipsalliance.cde.config.Config
import sattn._

class SattnRocketConfig extends Config(
  new WithSattnAccelerator() ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig
)

class SattnDebugConfig extends Config(
  new WithSattnAccelerator(
    sattn.SattnAcceleratorConfig(
      scratchpadKB = 4,
      indexRAMKB = 1,
      numPEs = 4,
      maxBlockSize = 32,
      maxSeqLen = 256
    )
  ) ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig
)
```

---

## Step 5: Build Verilator Simulator

```bash
cd chipyard/sims/verilator

# Build with sparse attention accelerator
# Use SattnDebugConfig for faster builds during development
make CONFIG=SattnDebugConfig

# Or for full configuration:
# make CONFIG=SattnRocketConfig

# This will take 1-2 hours
```

The output will be:
```
simulator-chipyard-SattnDebugConfig
```

---

## Step 6: Build Test Program

```bash
cd /path/to/riscv_extensions/hw/sim

# Compile test program
riscv64-unknown-elf-gcc \
    -march=rv64gc \
    -mabi=lp64d \
    -static \
    -I../runtime \
    -o test_hw_basic \
    test_hw_basic.c

# Convert to binary
riscv64-unknown-elf-objcopy -O binary test_hw_basic test_hw_basic.bin
```

---

## Step 7: Run Hardware Simulation

```bash
cd chipyard/sims/verilator

# Run test
./simulator-chipyard-SattnDebugConfig \
    +verbose \
    /path/to/test_hw_basic

# Expected output:
# =================================================================
# Hardware Sparse Attention Test
# =================================================================
# ...
# ✅ Hardware execution complete!
# Performance:
#   Cycles: XXXXX
#   Memory ops: XXXXX
# ...
# Test PASSED
# =================================================================
```

---

## Step 8: Run Python Validation

```bash
cd /path/to/riscv_extensions

# Run hardware vs software validation
python python/hardware_simulator.py \
    --simulator chipyard/sims/verilator/simulator-chipyard-SattnDebugConfig \
    --pattern sliding_window \
    --precision fp32 \
    --L 32 --D 16
```

---

## Troubleshooting

### Build Fails with Missing Dependencies

```bash
cd chipyard
./scripts/init-submodules-no-riscv-tools.sh
```

### Verilator Compilation is Slow

Use the debug configuration for faster builds:
```bash
make CONFIG=SattnDebugConfig
```

Or use parallel compilation:
```bash
make CONFIG=SattnDebugConfig -j$(nproc)
```

### Simulation Hangs

- Check that your test program is correct
- Add `+verbose` flag to see detailed trace
- Reduce problem size for debugging

---

## Performance Expectations

### SattnDebugConfig (Small, Fast)
- **Build Time**: 20-30 minutes
- **Sim Speed**: ~10 KHz (10K cycles/sec)
- **Max Sequence Length**: 256
- **Use Case**: Development, debugging

### SattnRocketConfig (Full Featured)
- **Build Time**: 1-2 hours
- **Sim Speed**: ~5 KHz
- **Max Sequence Length**: 2048
- **Use Case**: Performance validation

---

## Next Steps

1. **Validate Cycle Counts**: Compare hardware cycles to Phase 1 estimates
2. **Profile Memory Bandwidth**: Analyze memory access patterns
3. **Test All Patterns**: Run all 5 patterns × 4 precisions
4. **Optimize Hardware**: Tune scratchpad, PE count, etc.
5. **Generate Waveforms**: Use `+vcdfile=trace.vcd` for debugging

---

## Resources

- **Chipyard Docs**: https://chipyard.readthedocs.io/
- **Our Phase 3 Plan**: `hw/PHASE3_HARDWARE_PLAN.md`
- **RoCC Spec**: https://github.com/chipsalliance/rocket-chip/blob/master/docs/rocc.md
- **Our Hardware**: `hw/chisel/SparseAttentionAccelerator.scala`

---

## Quick Reference

```bash
# Build simulator
cd chipyard/sims/verilator && make CONFIG=SattnDebugConfig

# Run test
./simulator-chipyard-SattnDebugConfig /path/to/test_program

# Generate waveform
./simulator-chipyard-SattnDebugConfig +vcdfile=trace.vcd /path/to/test_program

# View waveform
gtkwave trace.vcd

# Clean build
make clean

# Rebuild from scratch
make CONFIG=SattnDebugConfig -B
```

