# Architecture Diagram: Multi-Backend Sparse Attention Library

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       User Application Code                          │
│              (Python / C / MLIR / PyTorch custom op)                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   Sparse Attention Library API                       │
│            Unified interface for multiple sparsity patterns          │
│  (sliding_window, block_topk, nm_structured, lsh, landmark, etc.)   │
└──────────┬────────────┬────────────────┬─────────────┬──────────────┘
           │            │                │             │
           ↓            ↓                ↓             ↓
    ┌──────────┐  ┌─────────┐   ┌──────────────┐  ┌──────────────────┐
    │   CPU    │  │   GPU   │   │     RVV      │  │   Custom ISA     │
    │ Backend  │  │ Backend │   │   Backend    │  │     Backend      │
    └──────────┘  └─────────┘   └──────────────┘  └──────────────────┘
         │             │                │                    │
         │             │                │                    │
         ↓             ↓                ↓                    ↓
    ┌──────────┐  ┌─────────┐   ┌──────────────┐  ┌──────────────────┐
    │  NumPy   │  │  Triton │   │   Standard   │  │ Custom opcode   │
    │  x86/ARM │  │  CUDA   │   │   RISC-V V   │  │    0x2B +       │
    │          │  │  ROCm   │   │  extension   │  │ CSRs 0x7C0-7DF  │
    └──────────┘  └─────────┘   └──────────────┘  └──────────────────┘
         ↓             ↓                ↓                    ↓
    ┌──────────┐  ┌─────────┐   ┌──────────────┐  ┌──────────────────┐
    │   Any    │  │  GPU    │   │     QEMU     │  │  Custom FPGA/    │
    │ Platform │  │Hardware │   │   -cpu rv64  │  │  ASIC Hardware   │
    │          │  │         │   │   ,v=true ✅ │  │  or QEMU plugin  │
    └──────────┘  └─────────┘   └──────────────┘  └──────────────────┘
```

## Backend Comparison Matrix

```
┌────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│   Feature      │     CPU      │     GPU      │     RVV      │  Custom ISA  │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ ISA Used       │  x86/ARM     │  CUDA PTX    │  RISC-V V    │  RISC-V +    │
│                │  standard    │              │  (standard)  │  custom 0x2B │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ QEMU Support   │  ✅ Native   │  N/A         │  ✅ -cpu     │  ⚠️ Plugin   │
│                │              │              │  rv64,v=true │  or patch    │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Portability    │  ✅ High     │  GPU only    │  ✅ RISC-V   │  Custom HW   │
│                │              │              │  with V ext  │  only        │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Performance    │  Baseline    │  High        │  Medium-High │  Maximum     │
│                │  (reference) │  throughput  │  vectorized  │  specialized │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Use Case       │  Testing     │  Training    │  Embedded    │  Edge/IoT    │
│                │  Reference   │  Inference   │  RISC-V SoC  │  custom accel│
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Status         │  ✅ Ready    │  ✅ Ready    │  ✅ Ready    │  ✅ Encodings│
│                │              │              │  Tested QEMU │  validated   │
└────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

## RISC-V Backend Deep Dive

```
┌──────────────────────────────────────────────────────────────────┐
│                    RISC-V Backends (2 Options)                   │
└────────────────────┬───────────────────────┬─────────────────────┘
                     │                       │
         ┌───────────▼─────────┐   ┌────────▼──────────────────────┐
         │   RVV Backend       │   │   Custom ISA Backend          │
         │   (Portable)        │   │   (Performance)               │
         └─────────────────────┘   └───────────────────────────────┘
                     │                       │
         ┌───────────▼─────────┐   ┌────────▼──────────────────────┐
         │ Standard Vector     │   │ Custom Instructions           │
         │ Instructions:       │   │ (7 primitives):               │
         │  • vle32.v          │   │  • blk_reduce   (0x0000002B) │
         │  • vse32.v          │   │  • topk_idx     (0x0000102B) │
         │  • vfmacc.vv        │   │  • gath2d       (0x0000202B) │
         │  • vfadd.vv         │   │  • scat2d       (0x0000302B) │
         │  • vfmax.vv         │   │  • spdot_bsr    (0x0000402B) │
         │  etc.               │   │  • softmax_fused(0x0000502B) │
         │                     │   │  • spmm_bsr     (0x0000602B) │
         └─────────────────────┘   └───────────────────────────────┘
                     │                       │
         ┌───────────▼─────────┐   ┌────────▼──────────────────────┐
         │ Compilation:        │   │ Compilation:                  │
         │ -march=rv64gcv      │   │ -DSATTN_USE_CSR_INSTRUCTIONS  │
         └─────────────────────┘   └───────────────────────────────┘
                     │                       │
         ┌───────────▼─────────┐   ┌────────▼──────────────────────┐
         │ QEMU Execution:     │   │ Execution:                    │
         │ qemu-riscv64 -cpu   │   │ • Custom FPGA/ASIC            │
         │ rv64,v=true ✅      │   │ • QEMU + plugin               │
         │                     │   │ • QEMU patched source         │
         │ WORKS NOW!          │   │ • Verilator simulation        │
         └─────────────────────┘   └───────────────────────────────┘
```

## Instruction Comparison

### RVV Backend (Standard Vector Instructions)

```assembly
# Example: Vector dot product (QK^T computation)
vsetvli  t0, a0, e32, m1, ta, ma    # Set vector length
vle32.v  v0, (a1)                   # Load Q vector
vle32.v  v1, (a2)                   # Load K vector
vfmul.vv v2, v0, v1                 # Element-wise multiply
vfredusum.vs v3, v2, v4             # Reduce sum
vse32.v  v3, (a3)                   # Store result
```

**Status**: ✅ These are standard instructions, work on any RISC-V with V extension

### Custom ISA Backend (Sparse Attention Accelerator)

```assembly
# Setup parameters via CSRs
li      t0, 0x7C0
csrw    t0, a0      # Write Q base address to CSR_SATTN_Q_BASE
li      t0, 0x7C1
csrw    t0, a1      # Write K base address to CSR_SATTN_K_BASE
# ... more CSR writes ...

# Issue custom instruction (QK^T sparse matmul)
.word   0x0000402B  # SATTN_SPDOT_BSR (opcode 0x2B, funct3=4)

# Issue softmax
.word   0x0000502B  # SATTN_SOFTMAX_FUSED

# Issue attention output (Attn × V)
.word   0x0000602B  # SATTN_SPMM_BSR
```

**Status**: ✅ Encodings validated, requires custom hardware or modified QEMU

## QEMU Support Matrix

```
┌─────────────────────┬─────────────┬────────────────────────────────┐
│      Backend        │ QEMU Status │           Command              │
├─────────────────────┼─────────────┼────────────────────────────────┤
│ RVV (Vector)        │ ✅ Works    │ qemu-riscv64 -cpu rv64,v=true  │
├─────────────────────┼─────────────┼────────────────────────────────┤
│ Custom ISA          │ ❌ Illegal  │ qemu-riscv64 program           │
│ (no plugin)         │ instruction │ (standard QEMU)                │
├─────────────────────┼─────────────┼────────────────────────────────┤
│ Custom ISA          │ ✅ Can work │ qemu-riscv64 -plugin           │
│ (with plugin)       │             │ libsattn_custom.so program     │
├─────────────────────┼─────────────┼────────────────────────────────┤
│ Custom ISA          │ ✅ Can work │ Modified QEMU with custom      │
│ (patched QEMU)      │             │ instruction support built-in   │
├─────────────────────┼─────────────┼────────────────────────────────┤
│ MMIO mode           │ ✅ Works    │ qemu-riscv64 program           │
│ (simulation)        │             │ (no -DSATTN_USE_CSR_INSTRUCTIONS)│
└─────────────────────┴─────────────┴────────────────────────────────┘
```

## Development Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                      Phase 1: Algorithm Dev                      │
│                      Use RVV Backend                             │
│                                                                  │
│  python scripts/build_and_run_rvv_qemu.py                       │
│                                                                  │
│  ✅ Works on standard QEMU                                      │
│  ✅ Fast iteration                                              │
│  ✅ Portable across RISC-V platforms                            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                  Phase 2: Hardware Design                        │
│                  Custom ISA Specification                        │
│                                                                  │
│  • Define instruction encodings  ✅ DONE                        │
│  • Design RTL (hw/rtl/*.sv)     ✅ Decoder ready               │
│  • Validate encodings            ✅ 53/53 tests passed         │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                Phase 3: Hardware Simulation                      │
│                Verilator / FPGA                                  │
│                                                                  │
│  • Verilator testbench (hw/sim/)                                │
│  • FPGA prototype                                               │
│  • Full functional verification                                 │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│              Phase 4: QEMU Plugin (Optional)                     │
│              For ISA-level testing                               │
│                                                                  │
│  • Create QEMU TCG plugin                                        │
│  • Emulate custom instructions                                   │
│  • Software testing without hardware                             │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                   Phase 5: Deployment                            │
│                   Real Hardware                                  │
│                                                                  │
│  • FPGA/ASIC with custom extension                              │
│  • Compile with -DSATTN_USE_CSR_INSTRUCTIONS                    │
│  • Maximum performance                                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Current Status (All Tests Passing!)

```
Test Suite                      Status       Command
─────────────────────────────────────────────────────────────────
RVV Backend on QEMU             ✅ PASS      python scripts/build_and_run_rvv_qemu.py
                                            Output: MATCH max_abs=3.91854e-07
                                                    checksum=532.774657
                                                    rvv_bytes_read=824320

Custom ISA Encoding Tests       ✅ PASS      qemu-riscv64 test_encoding_only
                                53/53        Output: All encoding validation
                                            tests passed!

Custom ISA Assembly Tests       ✅ PASS      Disassembly shows correct
                                            instruction words (0x2B opcode)

Custom ISA C Macros            ✅ PASS      Macros expand to correct
                                            instruction encodings
```

## Summary

✅ **Your library has a solid multi-backend architecture**

✅ **RVV backend works perfectly on QEMU** - use this for development

✅ **Custom ISA encodings are validated** - ready for hardware implementation

✅ **Unified API** - same code, different backends

🎯 **Next**: Implement hardware or continue with RVV for algorithm development

## See Also

- `docs/QEMU_GUIDE.md` - Quick start guide for QEMU
- `docs/backend_architecture.md` - Detailed architecture documentation
- `backends/rvv/README.md` - RVV backend documentation
- `hw/spec/instruction_encoding.md` - Custom ISA specification

