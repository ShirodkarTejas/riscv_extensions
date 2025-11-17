# Phase 3: Hardware Simulation - Implementation Plan

**Status**: 🚧 In Progress  
**Goal**: Validate software stack on cycle-accurate hardware simulation  
**Integration**: Builds on Phase 1 (validated) + Phase 2 (compiler)

---

## 🎯 Objectives

### Primary Goals
1. **Verilator Setup** - Cycle-accurate RTL simulation
2. **Chipyard Integration** - RISC-V SoC with custom accelerator
3. **Accelerator Implementation** - Sparse attention in hardware
4. **Co-Simulation** - Software + hardware validation
5. **Performance Validation** - Verify cycle counts, energy estimates

### Success Criteria
- ✅ Verilator simulation running
- ✅ Custom accelerator integrated with RISC-V core
- ✅ Generated code runs on simulated hardware
- ✅ Cycle counts match Phase 1 estimates
- ✅ Full-stack validation (Python → MLIR → C → Hardware)

---

## 📋 Phase 3 Tasks

### Task 1: Verilator Environment Setup (Days 1-2)

**Goal**: Get basic Verilator simulation working

#### 1.1 Install Dependencies
```bash
# Verilator
sudo apt-get install verilator

# Chipyard dependencies
sudo apt-get install build-essential bison flex \
    software-properties-common curl \
    default-jdk-headless sbt
```

#### 1.2 Setup Chipyard
```bash
# Clone Chipyard
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard
./build-setup.sh riscv-tools

# Build RISC-V tools
cd toolchains/riscv-tools/riscv-gnu-toolchain
./configure --prefix=$RISCV
make
```

#### 1.3 Basic Verilator Test
```bash
# Build a simple Rocket Chip config
cd chipyard/sims/verilator
make CONFIG=RocketConfig
./simulator-chipyard-RocketConfig +verbose
```

**Deliverable**: Working Verilator simulation of basic RISC-V core

---

### Task 2: Accelerator Architecture Design (Days 3-5)

**Goal**: Define hardware architecture for sparse attention

#### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RISC-V Core (Rocket/BOOM)                │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐  │
│  │   CPU    │   │    L1    │   │  RoCC Interface      │  │
│  │  Scalar  │──▶│  Cache   │──▶│  (Custom Coprocessor)│  │
│  └──────────┘   └──────────┘   └──────────┬───────────┘  │
└─────────────────────────────────────────────┼──────────────┘
                                              │
                            ┌─────────────────▼──────────────────┐
                            │  Sparse Attention Accelerator      │
                            │                                    │
                            │  ┌──────────┐  ┌──────────┐      │
                            │  │  Index   │  │  Scratchpad│      │
                            │  │   RAM    │  │   Memory   │      │
                            │  └──────────┘  └──────────┘      │
                            │                                    │
                            │  ┌──────────┐  ┌──────────┐      │
                            │  │  SPDOT   │  │  SOFTMAX │      │
                            │  │  Engine  │  │  Fused   │      │
                            │  └──────────┘  └──────────┘      │
                            │                                    │
                            │  ┌──────────┐  ┌──────────┐      │
                            │  │  GATHER  │  │  SPMM    │      │
                            │  │   2D     │  │  BSR     │      │
                            │  └──────────┘  └──────────┘      │
                            └────────────────────────────────────┘
```

#### 2.2 Hardware Primitives

Based on our existing RTL in `hw/rtl/`:

1. **`spdot_bsr_core.sv`** - Sparse dot product
2. **`softmax_fused_stub.sv`** - Fused softmax
3. **`spmm_bsr_stub.sv`** - Sparse matrix multiply
4. **`gather2d_stub.sv`** - 2D gather
5. **`idx_ram.sv`** - Index storage
6. **`spad.sv`** - Scratchpad memory
7. **`rocc_sattn.sv`** - RoCC interface wrapper

#### 2.3 Memory Hierarchy

```
CPU
 ↓
L1 Cache (32KB)
 ↓
RoCC Interface
 ↓
┌─────────────────────────────────────┐
│ Accelerator Local Memory            │
│                                     │
│  Scratchpad (16KB)                  │
│  ├─ Q tiles (4KB)                   │
│  ├─ K tiles (4KB)                   │
│  ├─ V tiles (4KB)                   │
│  └─ O tiles (4KB)                   │
│                                     │
│  Index RAM (2KB)                    │
│  └─ Sparsity patterns               │
└─────────────────────────────────────┘
 ↓
L2 Cache / Main Memory
```

#### 2.4 Instruction Set (RoCC)

```c
// RoCC Custom Instructions
CUSTOM_0: Load indices to accelerator
CUSTOM_1: Configure shape (B, H, L, D)
CUSTOM_2: Execute sparse attention
CUSTOM_3: Check status / get results
```

**Deliverable**: Complete architecture specification

---

### Task 3: Chisel/Verilog Implementation (Days 6-10)

**Goal**: Implement accelerator in synthesizable HDL

#### 3.1 Top-Level Module (Chisel)

```scala
// hw/chisel/SparseAttentionAccelerator.scala
package sattn

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import freechips.rocketchip.config._

class SparseAttentionAccelerator(implicit p: Parameters) extends LazyRoCC {
  override lazy val module = new SparseAttentionAcceleratorModule(this)
}

class SparseAttentionAcceleratorModule(outer: SparseAttentionAccelerator)
  extends LazyRoCCModuleImp(outer) {
  
  // State machine
  val idle :: load_indices :: config :: execute :: done :: Nil = Enum(5)
  val state = RegInit(idle)
  
  // Control registers
  val shape_B = Reg(UInt(32.W))
  val shape_H = Reg(UInt(32.W))
  val shape_L = Reg(UInt(32.W))
  val shape_D = Reg(UInt(32.W))
  
  // Pattern parameters
  val pattern = Reg(UInt(8.W))
  val window_size = Reg(UInt(32.W))
  
  // Datapath modules
  val spdot = Module(new SPDotBSR)
  val softmax = Module(new SoftmaxFused)
  val spmm = Module(new SPMMbsr)
  val gather = Module(new Gather2D)
  
  // Scratchpad memory
  val spad = Module(new Scratchpad(16*1024))
  val idx_ram = Module(new IndexRAM(2*1024))
  
  // RoCC interface handling
  when(io.cmd.fire()) {
    switch(io.cmd.bits.inst.funct) {
      is(0.U) { // CUSTOM_0: Load indices
        state := load_indices
      }
      is(1.U) { // CUSTOM_1: Configure
        shape_B := io.cmd.bits.rs1
        shape_H := io.cmd.bits.rs2
        state := config
      }
      is(2.U) { // CUSTOM_2: Execute
        state := execute
      }
      is(3.U) { // CUSTOM_3: Status
        io.resp.bits.data := Mux(state === done, 1.U, 0.U)
      }
    }
  }
  
  // State machine for execution
  when(state === execute) {
    // Implement sparse attention dataflow
    // 1. Load Q, K tiles
    // 2. Compute QK^T (sparse)
    // 3. Apply softmax
    // 4. Load V
    // 5. Compute attention * V
    // 6. Write output
    
    // ... (detailed implementation)
  }
}
```

#### 3.2 Integrate Existing Verilog Modules

We already have several Verilog modules in `hw/rtl/`:

```scala
// Import existing Verilog modules
class SPDotBSR extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val a_data = Input(Vec(8, UInt(32.W)))
    val b_data = Input(Vec(8, UInt(32.W)))
    val indices = Input(Vec(8, UInt(16.W)))
    val result = Output(UInt(32.W))
    val valid = Output(Bool())
  })
}

// Similar for other modules...
```

#### 3.3 Build System Integration

```scala
// hw/chisel/Configs.scala
class SattnRocketConfig extends Config(
  new WithSattnAccelerator ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig
)

class WithSattnAccelerator extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sattn = LazyModule(new SparseAttentionAccelerator()(p))
      sattn
    }
  )
})
```

**Deliverable**: Synthesizable Chisel/Verilog accelerator

---

### Task 4: Chipyard Integration (Days 11-13)

**Goal**: Integrate accelerator into Chipyard SoC

#### 4.1 Add to Chipyard Source

```bash
# Create generator directory
cd chipyard/generators
mkdir sattn
cp -r /path/to/riscv_extensions/hw/chisel/* sattn/

# Update build.sbt
echo 'lazy val sattn = (project in file("generators/sattn"))' >> build.sbt
```

#### 4.2 Update Chipyard Config

```scala
// chipyard/src/main/scala/config/RocketConfigs.scala
class SattnRocketConfig extends Config(
  new sattn.WithSattnAccelerator ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new freechips.rocketchip.subsystem.WithNBanks(1) ++
  new freechips.rocketchip.subsystem.WithL1ICacheWays(2) ++
  new chipyard.config.AbstractConfig
)
```

#### 4.3 Build with Verilator

```bash
cd chipyard/sims/verilator
make CONFIG=SattnRocketConfig
```

**Deliverable**: Verilator simulator with custom accelerator

---

### Task 5: Software Integration (Days 14-16)

**Goal**: Connect Phase 1/2 software to hardware simulator

#### 5.1 RoCC Driver Implementation

Update `hw/runtime/rocc_driver.h` to match hardware:

```c
// hw/runtime/rocc_driver.h
#ifndef ROCC_DRIVER_H
#define ROCC_DRIVER_H

#include <stdint.h>

// RoCC instruction macros
#define ROCC_INSTRUCTION_DSS(opcode, rd, rs1, rs2) \
    asm volatile ( \
        ".word (0x0b | (" #opcode " << 25) | (" #rd " << 7) | (" #rs1 " << 15) | (" #rs2 " << 20))" \
    )

// Sparse attention RoCC interface
static inline void sattn_rocc_load_indices(uint64_t addr, uint32_t size) {
    ROCC_INSTRUCTION_DSS(0, 0, addr, size);
}

static inline void sattn_rocc_configure(uint32_t B, uint32_t H, uint32_t L, uint32_t D) {
    ROCC_INSTRUCTION_DSS(1, 0, (B << 16) | H, (L << 16) | D);
}

static inline void sattn_rocc_execute(void) {
    ROCC_INSTRUCTION_DSS(2, 0, 0, 0);
}

static inline uint32_t sattn_rocc_status(void) {
    uint32_t status;
    ROCC_INSTRUCTION_DSS(3, status, 0, 0);
    return status;
}

// High-level sparse attention call
void sattn_hw_sliding_window(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int window_size
);

#endif
```

#### 5.2 Hardware Backend Implementation

```c
// backends/hw/src/sparse_attention_hw.c
#include "rocc_driver.h"
#include <string.h>

void sattn_hw_sliding_window(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int D,
    int window_size
) {
    // 1. Configure accelerator
    sattn_rocc_configure(B, H, L, D);
    
    // 2. Load sparsity indices (if needed)
    // Generate or load indices for sliding window pattern
    uint16_t* indices = generate_sliding_window_indices(L, window_size);
    sattn_rocc_load_indices((uint64_t)indices, L * sizeof(uint16_t));
    
    // 3. Copy Q, K, V to accelerator memory
    // (via DMA or memory-mapped interface)
    memcpy(HW_MEM_Q, Q, B*H*L*D*sizeof(float));
    memcpy(HW_MEM_K, K, B*H*L*D*sizeof(float));
    memcpy(HW_MEM_V, V, B*H*L*D*sizeof(float));
    
    // 4. Execute
    sattn_rocc_execute();
    
    // 5. Poll for completion
    while (sattn_rocc_status() == 0) {
        // Wait
    }
    
    // 6. Copy result
    memcpy(O, HW_MEM_O, B*H*L*D*sizeof(float));
}
```

#### 5.3 Test Harness

```c
// hw/sim/test_hw_backend.c
#include "sparse_attention_hw.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    int B=1, H=8, L=128, D=64;
    
    // Allocate tensors
    float *Q = malloc(B*H*L*D * sizeof(float));
    float *K = malloc(B*H*L*D * sizeof(float));
    float *V = malloc(B*H*L*D * sizeof(float));
    float *O = malloc(B*H*L*D * sizeof(float));
    
    // Initialize with test data
    for (int i = 0; i < B*H*L*D; i++) {
        Q[i] = (float)rand() / RAND_MAX;
        K[i] = (float)rand() / RAND_MAX;
        V[i] = (float)rand() / RAND_MAX;
    }
    
    // Run on hardware
    printf("Running sparse attention on hardware...\n");
    sattn_hw_sliding_window(Q, K, V, O, B, H, L, D, 16);
    printf("Done!\n");
    
    // Validate output
    // (compare with Phase 1 RVV reference)
    
    free(Q); free(K); free(V); free(O);
    return 0;
}
```

**Deliverable**: Software-hardware integration working

---

### Task 6: Co-Simulation Framework (Days 17-19)

**Goal**: Enable end-to-end validation (Python → Hardware)

#### 6.1 Python Hardware Interface

```python
# python/hardware_backend.py
import subprocess
import numpy as np
from pathlib import Path

class HardwareSimulator:
    """Interface to Verilator hardware simulation"""
    
    def __init__(self, verilator_binary: str):
        self.verilator_bin = verilator_binary
        
    def run_sparse_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        pattern: str,
        **kwargs
    ) -> np.ndarray:
        """Run sparse attention on hardware simulator"""
        
        # 1. Write inputs to files
        np.save('/tmp/Q.npy', Q)
        np.save('/tmp/K.npy', K)
        np.save('/tmp/V.npy', V)
        
        # 2. Run Verilator simulation
        result = subprocess.run([
            self.verilator_bin,
            '+pattern=' + pattern,
            '+Q=/tmp/Q.npy',
            '+K=/tmp/K.npy',
            '+V=/tmp/V.npy',
            '+O=/tmp/O.npy',
        ], capture_output=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Hardware simulation failed: {result.stderr}")
        
        # 3. Read output
        O = np.load('/tmp/O.npy')
        
        # 4. Parse performance metrics
        metrics = self._parse_verilator_output(result.stdout)
        
        return O, metrics
    
    def _parse_verilator_output(self, output: str) -> dict:
        """Extract cycle count, memory access, etc."""
        metrics = {}
        for line in output.split('\n'):
            if 'CYCLES:' in line:
                metrics['cycles'] = int(line.split(':')[1])
            if 'MEM_READS:' in line:
                metrics['mem_reads'] = int(line.split(':')[1])
            if 'MEM_WRITES:' in line:
                metrics['mem_writes'] = int(line.split(':')[1])
        return metrics
```

#### 6.2 End-to-End Validation Script

```python
# tests/test_hardware_e2e.py
import numpy as np
from python.hardware_backend import HardwareSimulator
from python.sparse_attention_rvv import sparse_attention_rvv
from validation.dense_attention_reference import dense_attention_reference

def test_hardware_vs_software():
    """Validate hardware simulation matches software"""
    
    # Test configuration
    B, H, L, D = 1, 2, 32, 16  # Small for fast testing
    pattern = "sliding_window"
    precision = "fp32"
    
    # Generate test data
    Q = np.random.randn(B, H, L, D).astype(np.float32)
    K = np.random.randn(B, H, L, D).astype(np.float32)
    V = np.random.randn(B, H, L, D).astype(np.float32)
    
    # 1. Run on hardware simulator
    hw_sim = HardwareSimulator('chipyard/sims/verilator/simulator-SattnRocketConfig')
    O_hw, hw_metrics = hw_sim.run_sparse_attention(Q, K, V, pattern, window_size=16)
    
    # 2. Run on software (Phase 1)
    O_sw = sparse_attention_rvv(Q, K, V, pattern=pattern, precision=precision, window_size=16)
    
    # 3. Run dense reference
    O_ref = dense_attention_reference(Q, K, V)
    
    # 4. Validate
    mae_hw_ref = np.mean(np.abs(O_hw - O_ref))
    mae_sw_ref = np.mean(np.abs(O_sw - O_ref))
    mae_hw_sw = np.mean(np.abs(O_hw - O_sw))
    
    print(f"Hardware vs Reference MAE: {mae_hw_ref:.6f}")
    print(f"Software vs Reference MAE: {mae_sw_ref:.6f}")
    print(f"Hardware vs Software MAE: {mae_hw_sw:.6f}")
    
    print(f"\nHardware Metrics:")
    print(f"  Cycles: {hw_metrics['cycles']}")
    print(f"  Memory Reads: {hw_metrics['mem_reads']}")
    print(f"  Memory Writes: {hw_metrics['mem_writes']}")
    
    # Hardware and software should match closely
    assert mae_hw_sw < 1e-3, "Hardware/software mismatch!"
    
    return hw_metrics

if __name__ == "__main__":
    metrics = test_hardware_vs_software()
    print("\n✅ Hardware simulation validated!")
```

**Deliverable**: End-to-end validation framework

---

### Task 7: Performance Validation (Days 20-21)

**Goal**: Verify cycle counts match Phase 1 estimates

#### 7.1 Cycle-Accurate Profiling

```python
# scripts/validate_hardware_performance.py
import pandas as pd
from python.hardware_backend import HardwareSimulator

def profile_hardware_vs_estimates():
    """Compare hardware cycle counts to Phase 1 estimates"""
    
    results = []
    
    patterns = ["sliding_window", "block_local_global", "nm_structured"]
    precisions = ["fp32", "bf16", "i8", "i4"]
    sizes = [32, 64, 128]
    
    hw_sim = HardwareSimulator('...')
    
    for pattern in patterns:
        for precision in precisions:
            for L in sizes:
                # Run on hardware
                Q = np.random.randn(1, 8, L, 64).astype(np.float32)
                K = np.random.randn(1, 8, L, 64).astype(np.float32)
                V = np.random.randn(1, 8, L, 64).astype(np.float32)
                
                _, hw_metrics = hw_sim.run_sparse_attention(
                    Q, K, V, pattern, precision=precision
                )
                
                # Get Phase 1 estimate
                sw_cycles = get_phase1_cycles(pattern, precision, L)
                
                # Compare
                results.append({
                    'pattern': pattern,
                    'precision': precision,
                    'L': L,
                    'hw_cycles': hw_metrics['cycles'],
                    'sw_estimate': sw_cycles,
                    'error_pct': abs(hw_metrics['cycles'] - sw_cycles) / sw_cycles * 100
                })
    
    df = pd.DataFrame(results)
    print(df)
    
    # Save report
    df.to_csv('bench/results/hardware_validation.csv', index=False)
    
    return df

if __name__ == "__main__":
    df = profile_hardware_vs_estimates()
    print(f"\nAverage error: {df['error_pct'].mean():.1f}%")
```

**Deliverable**: Performance validation report

---

## 🏗️ Implementation Strategy

### Week 1: Environment Setup
- **Days 1-2**: Verilator + Chipyard setup
- **Days 3-5**: Architecture design

### Week 2: Hardware Implementation
- **Days 6-10**: Chisel/Verilog coding
- **Days 11-13**: Chipyard integration

### Week 3: Software Integration & Validation
- **Days 14-16**: Software-hardware connection
- **Days 17-19**: Co-simulation framework
- **Days 20-21**: Performance validation

---

## 📊 Deliverables

### Code
1. **Accelerator HDL** - `hw/chisel/SparseAttentionAccelerator.scala`
2. **Hardware Backend** - `backends/hw/src/sparse_attention_hw.c`
3. **Python Interface** - `python/hardware_backend.py`
4. **Test Suite** - `tests/test_hardware_e2e.py`
5. **Validation Scripts** - `scripts/validate_hardware_performance.py`

### Documentation
1. **Architecture Spec** - `hw/ARCHITECTURE.md`
2. **Integration Guide** - `hw/CHIPYARD_INTEGRATION.md`
3. **User Guide** - `hw/HARDWARE_SIMULATION_GUIDE.md`
4. **Phase 3 Complete** - `hw/PHASE3_COMPLETE.md`

### Results
1. **Cycle counts** - Hardware vs software comparison
2. **Memory bandwidth** - Actual vs estimated
3. **Energy validation** - Match Phase 1 proxy model
4. **Accuracy validation** - Hardware vs software vs reference

---

## 🎯 Success Metrics

### Functional
- ✅ Hardware simulation completes without errors
- ✅ Output matches software (MAE < 1e-3)
- ✅ All 5 patterns work on hardware
- ✅ All 4 precisions supported

### Performance
- 🎯 Cycle counts within 10% of Phase 1 estimates
- 🎯 Memory bandwidth utilization > 80%
- 🎯 Accelerator speedup > 10× vs scalar RISC-V
- 🎯 Simulation time < 5 minutes for L=128

### Quality
- 🎯 100% test coverage
- 🎯 All validation tests passing
- 🎯 Complete documentation

---

## 🚀 Quick Start (After Implementation)

### Build Hardware Simulator
```bash
cd chipyard/sims/verilator
make CONFIG=SattnRocketConfig
```

### Run Test
```bash
./simulator-SattnRocketConfig \
    +pattern=sliding_window \
    +L=128 +D=64 \
    +verbose
```

### Validate End-to-End
```bash
python tests/test_hardware_e2e.py
```

---

## 📚 References

- **Chipyard**: https://chipyard.readthedocs.io/
- **Rocket Chip**: https://github.com/chipsalliance/rocket-chip
- **Verilator**: https://www.veripool.org/verilator/
- **Our RTL**: `hw/rtl/` (existing modules)
- **Phase 1 & 2**: Already complete!

---

**Status**: Ready to implement! Starting with Verilator setup...

