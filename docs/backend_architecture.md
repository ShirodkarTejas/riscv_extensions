# Backend Architecture and QEMU Support

This document explains the different backends in the sparse attention library and how they work with QEMU.

## Overview: Multi-Backend Architecture

The library provides **multiple backend implementations** of sparse attention primitives, allowing users to choose the best option for their target platform:

```
┌─────────────────────────────────────────────────────────────┐
│                   Sparse Attention Library                   │
│                     (Unified Python/C API)                   │
└────────┬──────────┬─────────────┬──────────────┬────────────┘
         │          │             │              │
    ┌────▼────┐ ┌──▼─────┐ ┌─────▼──────┐ ┌─────▼──────────┐
    │   CPU   │ │  GPU   │ │    RVV     │ │  Custom ISA    │
    │ (NumPy) │ │(Triton)│ │ (Vector)   │ │ (Accelerator)  │
    └────┬────┘ └──┬─────┘ └─────┬──────┘ └─────┬──────────┘
         │          │             │              │
    Standard    CUDA/ROCm   QEMU + RVV     Custom Hardware
```

## Backends Comparison

| Backend | ISA Extension | QEMU Support | Use Case | Status |
|---------|--------------|--------------|----------|--------|
| **CPU** | Standard | ✅ Native | Reference, testing | ✅ Ready |
| **GPU** | N/A (CUDA/Triton) | N/A | High throughput | ✅ Ready |
| **RVV** | Standard `V` (Vector) | ✅ `-cpu rv64,v=true` | Portable RISC-V | ✅ Ready |
| **Custom ISA** | Custom opcode 0x2B | ⚠️ Needs plugin/patch | Max performance | 🆕 Encodings ready |

## 1. RVV Backend (Portable RISC-V)

### What It Uses
- **ISA**: Standard RISC-V Vector (V) extension 1.0
- **Instructions**: Standard vector instructions (`vle`, `vse`, `vfmacc`, etc.)
- **Compilation**: `-march=rv64gcv`

### QEMU Support ✅
Works out of the box with standard QEMU:

```bash
qemu-riscv64 -L /usr/riscv64-linux-gnu \
             -cpu rv64,v=true,vlen=128,elen=64 \
             ./my_program
```

### Build and Run

```bash
# Build RVV backend
python scripts/build_and_run_rvv_qemu.py

# Run specific test
python scripts/build_and_run_rvv_qemu.py \
  --exe sattn_rvv_runner \
  --args "--spec sliding_window --L 128 --D 32 --window 8"

# Run with quantization
python scripts/build_and_run_rvv_qemu.py \
  --exe sattn_rvv_runner \
  --args "--spec block_local_global --L 128 --D 32 --precision i8"
```

### Example Output
```
MATCH max_abs=3.91854e-07
spec=sliding_window checksum=272.034012 rvv_bytes_read=398336 bytes_written=138240 mac_flops=32512
```

### When to Use RVV Backend
- ✅ Testing on QEMU without custom hardware
- ✅ Portable code for any RISC-V core with Vector extension
- ✅ Baseline performance measurements
- ✅ Algorithm development and verification

## 2. Custom Instruction Backend (Hardware Accelerator)

### What It Uses
- **ISA**: Custom instructions (opcode 0x2B)
- **CSRs**: Custom CSRs (0x7C0-0x7DF)
- **Instructions**: 7 primitives (blk_reduce, topk_idx, gath2d, scat2d, spdot_bsr, softmax_fused, spmm_bsr)
- **Compilation**: `-DSATTN_USE_CSR_INSTRUCTIONS`

### QEMU Support ⚠️
Standard QEMU **does not** recognize custom instructions. You have 3 options:

#### Option A: Use MMIO Mode (No QEMU Modification)
Run the API without executing custom instructions:

```bash
# Compile without custom instruction flag (uses MMIO simulation)
riscv64-linux-gnu-gcc -I. your_code.c -o program

# Run on standard QEMU
qemu-riscv64 ./program
```

This validates the API and encodings but doesn't execute the custom hardware.

#### Option B: QEMU TCG Plugin (Recommended for Testing)
Create a QEMU plugin that intercepts and emulates custom instructions:

```bash
# Build QEMU plugin (example structure)
gcc -shared -fPIC -o libsattn_custom.so \
    qemu_plugin/sattn_custom_insn.c \
    -I$QEMU_SOURCE/include

# Run with plugin
qemu-riscv64 -plugin libsattn_custom.so ./program
```

#### Option C: Patch QEMU Source (Full Integration)
Modify QEMU to support custom instructions natively:

1. Add instruction decoder in `target/riscv/insn32.decode`
2. Implement handlers in `target/riscv/insn_trans/`
3. Add CSR definitions to `target/riscv/csr.c`
4. Rebuild QEMU

See [QEMU Extension Guide](#qemu-extension-guide) below.

### When to Use Custom Instruction Backend
- ✅ Hardware with custom sparse attention accelerator
- ✅ Verilator/FPGA simulation with RTL
- ✅ Maximum performance (specialized hardware)
- ✅ Research on custom ISA extensions

## 3. Architecture Decision Flow

```
Do you have custom hardware? ─Yes→ Use Custom Instruction Backend
         │                              (compile with -DSATTN_USE_CSR_INSTRUCTIONS)
         No
         ↓
Running on RISC-V? ─Yes→ Use RVV Backend (works with QEMU)
         │
         No
         ↓
Have GPU? ─Yes→ Use GPU Backend (Triton/CUDA)
         │
         No
         ↓
Use CPU Backend (NumPy reference)
```

## QEMU Extension Guide

### Option 1: TCG Plugin (Easiest)

Create `qemu_plugin/sattn_custom_insn.c`:

```c
#include <qemu-plugin.h>
#include <stdint.h>

// Plugin to intercept custom instructions (opcode 0x2B)
void vcpu_insn_exec(unsigned int vcpu_index, void *userdata) {
    struct qemu_plugin_insn *insn = (struct qemu_plugin_insn *)userdata;
    uint32_t insn_word = qemu_plugin_insn_data(insn);
    
    uint8_t opcode = insn_word & 0x7F;
    if (opcode == 0x2B) {  // Custom opcode
        uint8_t funct3 = (insn_word >> 12) & 0x7;
        
        // Emulate instruction based on funct3
        switch (funct3) {
            case 0: emulate_blk_reduce(); break;
            case 1: emulate_topk_idx(); break;
            case 4: emulate_spdot_bsr(); break;
            // ... etc
        }
    }
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv) {
    qemu_plugin_register_vcpu_insn_exec_cb(NULL, vcpu_insn_exec, 
                                          QEMU_PLUGIN_CB_NO_REGS, NULL);
    return 0;
}
```

Build:
```bash
gcc -shared -fPIC -o libsattn_custom.so sattn_custom_insn.c \
    -I$QEMU_BUILD/include/qemu
```

### Option 2: QEMU Source Patch (Full Support)

#### Step 1: Add Instruction Decoder

Edit `target/riscv/insn32.decode`:

```
# Sparse Attention custom instructions
sattn_blk_reduce  0000000 ..... ..... 000 ..... 0101011 @r
sattn_topk_idx    0000000 ..... ..... 001 ..... 0101011 @r
sattn_gath2d      0000000 ..... ..... 010 ..... 0101011 @r
sattn_scat2d      0000000 ..... ..... 011 ..... 0101011 @r
sattn_spdot_bsr   0000000 ..... ..... 100 ..... 0101011 @r
sattn_softmax     0000000 ..... ..... 101 ..... 0101011 @r
sattn_spmm_bsr    0000000 ..... ..... 110 ..... 0101011 @r
```

#### Step 2: Implement Handlers

Create `target/riscv/insn_trans/trans_sattn.c.inc`:

```c
static bool trans_sattn_spdot_bsr(DisasContext *ctx, arg_r *a) {
    // Read CSRs to get parameters
    TCGv q_base = tcg_temp_new();
    tcg_gen_ld_tl(q_base, cpu_env, offsetof(CPURISCVState, csr_sattn_q_base));
    
    // Call helper function
    gen_helper_sattn_spdot_bsr(cpu_env, q_base, /* ... */);
    
    tcg_temp_free(q_base);
    return true;
}
```

#### Step 3: Add Custom CSRs

Edit `target/riscv/csr.c`:

```c
static const target_ulong sattn_csr_list[] = {
    [CSR_SATTN_Q_BASE]    = 0x7C0,
    [CSR_SATTN_K_BASE]    = 0x7C1,
    // ... rest of CSRs
};

static RISCVException read_sattn_csr(CPURISCVState *env, int csrno,
                                     target_ulong *val) {
    *val = env->csr_sattn[csrno - CSR_SATTN_BASE];
    return RISCV_EXCP_NONE;
}

static RISCVException write_sattn_csr(CPURISCVState *env, int csrno,
                                      target_ulong val) {
    env->csr_sattn[csrno - CSR_SATTN_BASE] = val;
    return RISCV_EXCP_NONE;
}
```

#### Step 4: Rebuild QEMU

```bash
cd $QEMU_SOURCE
./configure --target-list=riscv64-softmmu,riscv64-linux-user
make -j$(nproc)
sudo make install
```

## Testing Matrix

| Test Type | Backend | QEMU Command | Status |
|-----------|---------|--------------|--------|
| Encoding validation | Custom ISA | Standard QEMU | ✅ Passed (53/53) |
| RVV execution | RVV | `qemu-riscv64 -cpu rv64,v=true` | ✅ Works |
| Custom ISA execution | Custom ISA | Standard QEMU | ❌ Illegal instruction |
| Custom ISA with plugin | Custom ISA | `qemu-riscv64 -plugin ...` | ⚠️ Requires plugin |
| MMIO simulation | Custom ISA (MMIO mode) | Standard QEMU | ✅ Works |

## Recommended Workflow

### Phase 1: Algorithm Development (NOW)
```bash
# Use RVV backend for portable testing
python scripts/build_and_run_rvv_qemu.py
```
✅ Works with standard QEMU!

### Phase 2: Custom Hardware Design (NEXT)
```bash
# Validate custom instruction encodings (already done!)
cd backends/rocc/tests
make -f Makefile_instruction_tests test_encoding_only
qemu-riscv64 ./test_encoding_only
```
✅ Encodings verified!

### Phase 3: Hardware Simulation
```bash
# Verilator simulation with RTL
cd hw/sim
./verilator_tb --test custom_instructions
```

### Phase 4: QEMU Plugin (Optional)
```bash
# Create plugin for instruction emulation
# Allows testing custom ISA API without hardware
```

### Phase 5: Real Hardware
```bash
# Deploy to FPGA/ASIC with custom extension
# Run with -DSATTN_USE_CSR_INSTRUCTIONS
```

## Summary

### What Works Now ✅
- **RVV backend**: Full QEMU support, portable, ready for testing
- **Custom instruction encodings**: Validated and correct
- **API layer**: Compiles and works in both modes

### What Needs Custom QEMU ⚠️
- Executing custom instructions (opcode 0x2B)
- Reading/writing custom CSRs (0x7C0-0x7DF)

### Recommendation
**Use RVV backend for development and testing** - it works perfectly with QEMU and provides a portable baseline. The custom instruction backend is for when you have dedicated hardware.

Your library is **architected correctly** to support both paths!

## See Also
- `backends/rvv/README.md` - RVV backend documentation
- `hw/spec/instruction_encoding.md` - Custom instruction specification
- `backends/rocc/tests/TEST_RESULTS.md` - Test results
- QEMU plugin documentation: https://qemu.readthedocs.io/en/latest/devel/tcg-plugins.html

