# Chipyard Setup in Docker - Complete Guide

This guide explains how to set up and run Chipyard hardware simulation inside our Docker container.

---

## 🎯 Overview

We've created a specialized Docker environment that includes:
- **Chipyard**: RISC-V SoC generator framework
- **Verilator**: Fast cycle-accurate simulator
- **RISC-V Toolchain**: Cross-compiler for test programs
- **Our Accelerator**: Sparse attention hardware implementation

**Total Setup Time**: 2-3 hours (mostly automated)

---

## 📋 Prerequisites

- Docker installed and running
- 16+ GB RAM available
- 50+ GB disk space
- Internet connection (for downloading Chipyard)

---

## 🚀 Quick Start

### Step 1: Build the Docker Image

```bash
cd /Users/tsh/do/riscv_extensions

# Build/rebuild the Docker image with Chipyard dependencies
# (Same container you've been using, now with Chipyard support!)
docker-compose -f docker/docker-compose.yml build

# This takes 10-15 minutes
```

### Step 2: Start the Container

```bash
# Start the container in the background
docker-compose -f docker/docker-compose.yml up -d

# Or if already running from previous work, just use it!
```

### Step 3: Enter the Container

```bash
# Same container as always:
docker exec -it sattn_rvv_dev bash

# You should now be inside the container at /workspace
```

### Step 4: Run Automated Setup

```bash
# Inside the container, run the setup script
bash scripts/setup_chipyard_in_docker.sh

# This will:
# 1. Clone Chipyard (~5 min)
# 2. Initialize submodules (~15 min)
# 3. Copy our accelerator code (~1 min)
# 4. Build Verilator simulator (~45 min)
# 5. Compile test program (~1 min)
# 6. Run validation test (~5 min)
#
# Total: ~70 minutes (mostly automated)
```

**Go get coffee! ☕ This will take about an hour.**

---

## 📊 What Happens During Setup

### Phase 1: Chipyard Clone
```
Cloning https://github.com/ucb-bar/chipyard.git
Checking out stable version 1.10.0
✅ ~300 MB download
```

### Phase 2: Submodule Initialization
```
Initializing Rocket Chip, BOOM, Verilator tools
Downloading RISC-V tools
✅ ~2 GB download
```

### Phase 3: Accelerator Integration
```
Copying Chisel code to generators/sattn/
Copying Verilog RTL modules
Updating build.sbt configuration
✅ Accelerator integrated
```

### Phase 4: Verilator Build (LONGEST STEP)
```
Generating Verilog from Chisel
Compiling with Verilator (C++ backend)
Linking simulator executable
✅ simulator-chipyard-SattnDebugConfig created
```

### Phase 5: Test Compilation
```
Cross-compiling test_hw_basic.c
Linking with RoCC driver
✅ RISC-V binary ready
```

### Phase 6: Validation
```
Running simple test on simulator
Checking basic functionality
✅ Hardware simulation working!
```

---

## 🧪 Running Tests

### Basic Test

```bash
cd /workspace/chipyard/sims/verilator

# Run the test program
./simulator-chipyard-SattnDebugConfig \
    +max-cycles=100000 \
    +verbose \
    /workspace/hw/sim/test_hw_basic
```

**Expected output**:
```
=================================================================
Hardware Sparse Attention Test
=================================================================
Configuration:
  Batch (B):     1
  Heads (H):     2
  Seq Len (L):   32
  Head Dim (D):  16
...
✅ Hardware execution complete!
Performance:
  Cycles: XXXXX
  Memory ops: XXXXX
...
Test PASSED
=================================================================
```

### Generate Waveforms

```bash
# Run with VCD output
./simulator-chipyard-SattnDebugConfig \
    +vcdfile=trace.vcd \
    +max-cycles=100000 \
    /workspace/hw/sim/test_hw_basic

# View waveform (if X11 forwarding is set up)
gtkwave trace.vcd
```

### Python Interface

```python
# Inside the container
python3 /workspace/python/hardware_simulator.py

# Or run validation
python3 << 'EOF'
from hardware_simulator import HardwareSimulator
import numpy as np

# Initialize simulator
hw_sim = HardwareSimulator(
    '/workspace/chipyard/sims/verilator/simulator-chipyard-SattnDebugConfig'
)

# Generate test data
Q = np.random.randn(1, 2, 32, 16).astype(np.float32)
K = np.random.randn(1, 2, 32, 16).astype(np.float32)
V = np.random.randn(1, 2, 32, 16).astype(np.float32)

# Run on hardware
O, metrics = hw_sim.run(Q, K, V, pattern='sliding_window', window_size=8)

print(f"✅ Cycles: {metrics['cycles']}")
print(f"✅ Memory ops: {metrics['mem_reads'] + metrics['mem_writes']}")
EOF
```

---

## 🔧 Configuration Options

### SattnDebugConfig (Default - Fast)
- **Scratchpad**: 4 KB
- **PEs**: 4
- **Max Seq Len**: 256
- **Build Time**: 30-45 min
- **Sim Speed**: ~10 KHz
- **Use Case**: Development, debugging

### SattnRocketConfig (Full Featured)
- **Scratchpad**: 16 KB
- **PEs**: 8
- **Max Seq Len**: 2048
- **Build Time**: 1-2 hours
- **Sim Speed**: ~5 KHz
- **Use Case**: Performance validation

To build full config:
```bash
cd /workspace/chipyard/sims/verilator
make CONFIG=SattnRocketConfig -j$(nproc)
```

---

## 🐛 Troubleshooting

### Build Fails: Out of Memory

**Problem**: Verilator compilation runs out of memory

**Solution**: Reduce parallelism
```bash
make CONFIG=SattnDebugConfig -j2  # Instead of -j$(nproc)
```

Or increase Docker memory limit in Docker Desktop:
- Settings → Resources → Memory → 8+ GB

### Build Fails: Java Heap Space

**Problem**: SBT runs out of heap space

**Solution**: Already set in docker-compose:
```yaml
environment:
  - SBT_OPTS=-Xmx4G -XX:+UseG1GC
```

If still failing, increase to 6G or 8G in `docker-compose.chipyard.yml`

### Simulation Hangs

**Problem**: Simulator appears stuck

**Solution 1**: Add timeout
```bash
timeout 300 ./simulator-chipyard-SattnDebugConfig ...
```

**Solution 2**: Add max cycles
```bash
./simulator-chipyard-SattnDebugConfig +max-cycles=10000 ...
```

**Solution 3**: Enable verbose mode
```bash
./simulator-chipyard-SattnDebugConfig +verbose ...
```

### Container Runs Out of Space

**Problem**: /workspace fills up

**Solution**: Clean Chipyard build artifacts
```bash
cd /workspace/chipyard
make clean

# Or clean everything
cd /workspace/chipyard/sims/verilator
make cleanall
```

---

## 📁 File Locations (Inside Container)

```
/workspace/
├── chipyard/                           # Chipyard framework
│   ├── generators/sattn/               # Our accelerator (copied)
│   ├── sims/verilator/                # Verilator simulator
│   │   └── simulator-chipyard-SattnDebugConfig  # Binary
│   └── src/main/scala/config/
│       └── SattnConfigs.scala         # Our configs
├── hw/
│   ├── chisel/                        # Source Chisel code
│   ├── runtime/rocc_driver_hw.h       # Hardware driver API
│   └── sim/test_hw_basic.c            # Test program
├── python/
│   └── hardware_simulator.py          # Python interface
└── scripts/
    └── setup_chipyard_in_docker.sh    # Setup script
```

---

## 💡 Tips & Best Practices

### 1. Use Persistent Volumes

The docker-compose file creates a persistent volume for Chipyard:
```yaml
volumes:
  - chipyard_data:/workspace/chipyard
```

This means Chipyard persists between container restarts!

### 2. Incremental Builds

After initial setup, rebuilds are much faster:
```bash
# Only rebuild if you change Chisel code
cd /workspace/chipyard/sims/verilator
make CONFIG=SattnDebugConfig

# Takes ~5-10 minutes instead of 45
```

### 3. Parallel Simulation

Run multiple tests in parallel:
```bash
# Terminal 1
./simulator-chipyard-SattnDebugConfig test1 &

# Terminal 2
./simulator-chipyard-SattnDebugConfig test2 &
```

### 4. Save Simulator Binary

The simulator binary is standalone:
```bash
# Copy out of container for later use
docker cp sattn_chipyard_dev:/workspace/chipyard/sims/verilator/simulator-chipyard-SattnDebugConfig .
```

---

## 📊 Resource Usage

### During Build
- **CPU**: 100% (all cores)
- **Memory**: 4-8 GB
- **Disk**: +5 GB

### During Simulation
- **CPU**: 1 core at 100%
- **Memory**: 1-2 GB
- **Disk**: Minimal (VCD files can be large)

---

## 🔄 Starting Over

If something goes wrong, you can start fresh:

```bash
# Exit container
exit

# Stop and remove container
docker-compose -f docker/docker-compose.chipyard.yml down

# Remove persistent volume (WARNING: deletes Chipyard)
docker volume rm riscv_extensions_chipyard_data

# Start over
docker-compose -f docker/docker-compose.chipyard.yml up -d
docker exec -it sattn_chipyard_dev bash
bash scripts/setup_chipyard_in_docker.sh
```

---

## ✅ Verification Checklist

After setup completes, verify:

- [ ] Simulator binary exists:
  ```bash
  ls -lh /workspace/chipyard/sims/verilator/simulator-chipyard-SattnDebugConfig
  ```

- [ ] Test program compiled:
  ```bash
  ls -lh /workspace/hw/sim/test_hw_basic
  file /workspace/hw/sim/test_hw_basic  # Should be RISC-V binary
  ```

- [ ] Simulator runs:
  ```bash
  /workspace/chipyard/sims/verilator/simulator-chipyard-SattnDebugConfig --help
  ```

- [ ] Python interface works:
  ```bash
  python3 -c "from hardware_simulator import HardwareSimulator; print('✅ OK')"
  ```

---

## 🚀 Next Steps

Once setup is complete:

1. **Run comprehensive tests**: Test all 20 configurations
2. **Profile performance**: Compare with Phase 1 estimates
3. **Generate waveforms**: Debug with GTKWave
4. **Optimize parameters**: Tune scratchpad, PE count
5. **Prepare for Phase 4**: FPGA deployment

---

## 📞 Support

### Documentation
- This guide: `hw/DOCKER_CHIPYARD_SETUP.md`
- Chipyard docs: https://chipyard.readthedocs.io/
- Our Phase 3 plan: `hw/PHASE3_HARDWARE_PLAN.md`

### Common Issues
- Out of memory → Reduce parallelism or increase Docker memory
- Build fails → Check logs, clean and rebuild
- Simulation hangs → Add +max-cycles flag

---

**Ready to begin?** Run:
```bash
docker-compose -f docker/docker-compose.chipyard.yml up -d
docker exec -it sattn_chipyard_dev bash
bash scripts/setup_chipyard_in_docker.sh
```

Then wait ~70 minutes for the magic to happen! ✨

