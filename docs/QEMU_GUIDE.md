# Quick Guide: Running on QEMU

## TL;DR - What Works Now

✅ **RVV Backend works perfectly with standard QEMU!**

```bash
# Inside Docker container
python scripts/build_and_run_rvv_qemu.py
```

That's it! No QEMU modifications needed for the RVV backend.

---

## Understanding the Architecture

You have **2 RISC-V backends** in your library:

### Backend 1: RVV (Standard Vector Extension) ✅
- **Uses**: Standard RISC-V Vector instructions (`vle`, `vse`, `vfmacc`, etc.)
- **QEMU**: Works with `-cpu rv64,v=true` (no modifications)
- **Purpose**: Portable across any RISC-V core with V extension
- **Status**: ✅ **Working and tested**

### Backend 2: Custom Instructions (Hardware Accelerator) 🆕
- **Uses**: Custom opcode 0x2B + CSRs 0x7C0-0x7DF
- **QEMU**: Needs plugin or source modification
- **Purpose**: Maximum performance with dedicated hardware
- **Status**: ✅ Encodings validated, execution needs hardware

**They are independent!** You pick one based on your target platform.

---

## Quick Start: RVV Backend (Recommended for Testing)

### 1. Build and Run

```bash
# Start Docker
cd /path/to/riscv_extensions
docker-compose -f docker/docker-compose.yml up -d

# Run RVV tests
docker exec sattn_rvv_dev bash -c \
  "python3 scripts/build_and_run_rvv_qemu.py"

# Run with custom parameters
docker exec sattn_rvv_dev bash -c \
  "python3 scripts/build_and_run_rvv_qemu.py \
   --exe sattn_rvv_runner \
   --args '--spec sliding_window --L 256 --D 64 --window 16'"
```

### 2. Expected Output

```
[run] cmake -G Ninja -S backends/rvv -B build/rvv-riscv64 ...
[run] cmake --build build/rvv-riscv64 -j
[run] qemu-riscv64 -L /usr/riscv64-linux-gnu \
      -cpu rv64,v=true,vlen=128,elen=64 \
      build/rvv-riscv64/sattn_rvv_compare_sw
MATCH max_abs=3.91854e-07
```

✅ **Success!** Your RVV code runs on QEMU.

### 3. Run Different Specs

```bash
# Sliding window pattern
python scripts/build_and_run_rvv_qemu.py \
  --exe sattn_rvv_runner \
  --args "--spec sliding_window --L 128 --D 32 --window 8"

# Block-local-global (topk) pattern
python scripts/build_and_run_rvv_qemu.py \
  --exe sattn_rvv_runner \
  --args "--spec block_local_global --L 128 --D 32 --block_size 16"

# With quantization (int8)
python scripts/build_and_run_rvv_qemu.py \
  --exe sattn_rvv_runner \
  --args "--spec sliding_window --L 128 --D 32 --precision i8"
```

---

## Custom Instructions on QEMU (Advanced)

The **custom instruction backend** needs special QEMU support. You have 3 options:

### Option 1: MMIO Mode (No QEMU Changes)

Compile without custom instructions - uses memory-mapped I/O simulation:

```bash
# Default compilation (no -DSATTN_USE_CSR_INSTRUCTIONS)
riscv64-linux-gnu-gcc -I. your_code.c -o program

# Runs on standard QEMU
qemu-riscv64 ./program
```

**Limitation**: Doesn't execute custom instructions, only validates API.

### Option 2: QEMU TCG Plugin (Recommended for Testing)

Create a plugin to emulate custom instructions:

```bash
# Plugin intercepts opcode 0x2B and emulates instructions
qemu-riscv64 -plugin /path/to/libsattn_custom.so ./program
```

**Advantage**: No QEMU source modification needed.

See `docs/backend_architecture.md` for plugin creation guide.

### Option 3: Patch QEMU Source (Full Integration)

Modify QEMU to natively support custom instructions:

1. Add decoder to `target/riscv/insn32.decode`
2. Implement handlers in `target/riscv/insn_trans/`
3. Add CSRs to `target/riscv/csr.c`
4. Rebuild QEMU

**Advantage**: Full integration, best for long-term development.

See `docs/backend_architecture.md` for detailed instructions.

---

## Decision Matrix

```
┌─────────────────────────────────────────────────────────┐
│ What should I use for...                                 │
├─────────────────────────────────────────────────────────┤
│ Testing on QEMU?              → RVV Backend ✅           │
│ Algorithm development?         → RVV Backend ✅           │
│ Portable RISC-V deployment?    → RVV Backend ✅           │
│ Custom hardware accelerator?   → Custom ISA Backend 🆕   │
│ Maximum performance?           → Custom ISA Backend 🆕   │
│ FPGA/ASIC implementation?      → Custom ISA Backend 🆕   │
└─────────────────────────────────────────────────────────┘
```

---

## Relationship Between Backends

```
Your Code
    ↓
┌───────────────────────────────────────────┐
│      Sparse Attention Library API          │
└───────────────────────────────────────────┘
    ↓                            ↓
┌──────────────┐      ┌──────────────────────┐
│ RVV Backend  │      │ Custom ISA Backend   │
│ (Standard V) │      │ (Opcode 0x2B)        │
└──────────────┘      └──────────────────────┘
    ↓                            ↓
┌──────────────┐      ┌──────────────────────┐
│ QEMU         │      │ Custom Hardware      │
│ -cpu rv64,v  │      │ or Modified QEMU     │
└──────────────┘      └──────────────────────┘
    ✅                          🆕
Works now!              Encodings ready!
```

---

## Testing Status

| Component | QEMU Support | Status |
|-----------|--------------|--------|
| **RVV Backend** | ✅ Standard QEMU | ✅ Tested, working |
| **Custom ISA Encodings** | ✅ Validation only | ✅ 53/53 tests passed |
| **Custom ISA Execution** | ⚠️ Needs plugin/patch | 🆕 Requires hardware |

---

## Example Workflow

### Phase 1: Development (Use RVV)

```bash
# Develop and test with RVV backend
python scripts/build_and_run_rvv_qemu.py

# Run autotune
python scripts/rvv_autotune_sweep.py \
  --spec sliding_window --L 256 --D 64 --qemu
```

### Phase 2: Validate Custom ISA Encodings

```bash
# Test instruction encodings
docker exec sattn_rvv_dev bash -c \
  "cd /workspace/backends/rocc/tests && \
   riscv64-linux-gnu-gcc -O2 -static \
   test_encoding_only.c -o test_encoding && \
   qemu-riscv64 ./test_encoding"

# Output: 53/53 tests passed ✅
```

### Phase 3: Deploy to Hardware

```bash
# Compile for custom hardware
riscv64-linux-gnu-gcc -DSATTN_USE_CSR_INSTRUCTIONS \
  -I. your_code.c -o program

# Run on FPGA/ASIC with custom extension
./program
```

---

## Summary

✅ **For QEMU**: Use the **RVV backend** - it works perfectly!

🆕 **Custom instructions**: Encodings are validated and ready. Execution requires:
- Hardware with custom extension, OR
- QEMU plugin, OR  
- QEMU source modification

**Your library is properly architected** to support both paths. Users can choose based on their target platform.

---

## Next Steps

1. ✅ **Continue development with RVV backend** (works on QEMU)
2. 🆕 **Design custom hardware** using the validated instruction encodings
3. ⚠️ **Optional**: Create QEMU plugin for custom instruction testing
4. 🎯 **Deploy**: Choose backend based on target (RVV for portable, Custom ISA for performance)

---

## Files to Reference

- `backends/rvv/README.md` - RVV backend full documentation
- `docs/backend_architecture.md` - Complete architecture guide
- `hw/spec/instruction_encoding.md` - Custom instruction specification
- `backends/rocc/tests/TEST_RESULTS.md` - Test results and validation
- `examples/backend_selection_demo.c` - Code example using both backends

---

## Questions?

**Q: Why did my custom instruction test fail on QEMU?**  
A: Standard QEMU doesn't recognize custom instructions. Use the RVV backend for testing, or create a QEMU plugin.

**Q: Do I need custom instructions?**  
A: Only if you're building custom hardware. The RVV backend provides a portable, performant baseline.

**Q: Can I use both backends?**  
A: Yes! Compile-time flag selects which backend to use. Same API, different implementation.

**Q: Will QEMU ever support my custom instructions natively?**  
A: Not in standard QEMU (custom extensions are project-specific). Use plugins or maintain a fork.

