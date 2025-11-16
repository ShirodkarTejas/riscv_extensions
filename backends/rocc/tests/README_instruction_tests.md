# Custom Instruction Test Suite

This directory contains tests for the custom RISC-V instruction encoding and CSR interface for sparse attention primitives.

## Test Files

### C Tests
- **`test_instruction_encoding.c`** - Comprehensive C test suite covering:
  - CSR read/write functionality
  - Hardware capability detection
  - Status register operations
  - Instruction encoding verification
  - Parameter setup helpers
  - Descriptor-based API
  - Error handling

### Assembly Tests
- **`test_instruction_asm.S`** - Low-level assembly tests:
  - Basic CSR access patterns
  - Instruction issue sequences
  - Complete attention pipeline
  - Encoding verification
  - All primitives (blk_reduce, topk_idx, gath2d, scat2d, spdot_bsr, softmax_fused, spmm_bsr)

## Building Tests

### Prerequisites

```bash
# RISC-V toolchain (GCC or LLVM)
export RISCV=/path/to/riscv/toolchain
export PATH=$RISCV/bin:$PATH

# Verify toolchain
riscv64-unknown-elf-gcc --version
```

### Compile C Tests

```bash
# Without custom instructions (MMIO mode, for simulation)
riscv64-unknown-elf-gcc -march=rv64gc -O2 \
  -I../../../ \
  -o test_instruction_encoding \
  test_instruction_encoding.c

# With custom instructions (CSR mode, for hardware)
riscv64-unknown-elf-gcc -march=rv64gc -O2 \
  -DSATTN_USE_CSR_INSTRUCTIONS \
  -I../../../ \
  -o test_instruction_encoding_csr \
  test_instruction_encoding.c
```

### Compile Assembly Tests

```bash
# Compile assembly
riscv64-unknown-elf-as -march=rv64gc \
  -o test_instruction_asm.o \
  test_instruction_asm.S

# Create test program (requires main wrapper)
riscv64-unknown-elf-gcc -march=rv64gc \
  test_instruction_asm.o \
  -o test_instruction_asm \
  -nostartfiles

# Or link with C test for combined suite
riscv64-unknown-elf-gcc -march=rv64gc -O2 \
  -I../../../ \
  test_instruction_encoding.c test_instruction_asm.S \
  -o test_combined
```

## Running Tests

### On Spike (RISC-V ISA Simulator)

```bash
# Run with Spike (may need custom extension plugin)
spike --isa=rv64gc pk test_instruction_encoding

# With debug output
spike -l --isa=rv64gc pk test_instruction_encoding
```

### On QEMU

```bash
# Standard QEMU (MMIO mode only)
qemu-riscv64 test_instruction_encoding

# QEMU with custom extension (if available)
qemu-riscv64 -cpu rv64,x-sattn=true test_instruction_encoding_csr
```

### On Hardware

```bash
# Copy to target board
scp test_instruction_encoding_csr user@board:/tmp/

# Run on hardware
ssh user@board /tmp/test_instruction_encoding_csr
```

### Expected Output

```
========================================
Custom RISC-V Instruction Test Suite
========================================

Mode: CSR-based instruction interface

Testing CSR read/write...
PASS: test_csr_read_write
Testing hardware capability detection...
  Hardware version: 0x00010000
  Capabilities: 0x0000000f
  - BSR format supported
  - BF16 precision supported
PASS: test_hw_capabilities
Testing status register...
PASS: test_status_register
...

========================================
Test Results:
  Passed: 8
  Failed: 0
========================================
```

## Test Coverage

### Functionality Tested

1. **CSR Operations**
   - 32-bit read/write
   - 64-bit pointer read/write (RV32/RV64 compatibility)
   - Float value read/write
   - CSR address validation

2. **Instruction Encoding**
   - Opcode verification (0x2B)
   - funct3 field mapping (7 primitives)
   - Encoding macro correctness
   - Instruction word generation

3. **Status and Control**
   - READY, BUSY, DONE bit transitions
   - Error flag detection
   - Hardware capability querying
   - Version detection

4. **High-Level APIs**
   - Descriptor-based interface
   - Bulk parameter setup
   - Unified API (compile-time mode selection)

5. **Assembly Primitives**
   - All 7 instructions issue correctly
   - CSR staging patterns
   - Polling loops
   - Complete pipeline execution

## Debugging Tests

### Check Instruction Encoding

```bash
# Disassemble to verify instruction encodings
riscv64-unknown-elf-objdump -d test_instruction_asm.o | grep -A2 "\.word"

# Expected output:
#   0:  0000002b    .word 0x0000002b  # blk_reduce
#   4:  0000102b    .word 0x0000102b  # topk_idx
#   ...
```

### GDB Debugging

```bash
# Start GDB with Spike
spike --isa=rv64gc -d pk test_instruction_encoding

# In another terminal
riscv64-unknown-elf-gdb test_instruction_encoding
(gdb) target remote :9824
(gdb) break test_csr_read_write
(gdb) continue
(gdb) info registers
(gdb) x/10i $pc
```

### View CSR State

```bash
# In GDB, read CSRs manually
(gdb) print/x $csr0x7C0  # Q_BASE
(gdb) print/x $csr0x7CE  # STATUS
```

## Integration with Hardware

### Verilator Simulation

```bash
# Assuming Verilator testbench in hw/sim/
cd ../../../hw/sim
./run_from_mlir.py --test-asm ../../backends/rocc/tests/test_instruction_asm.S
```

### Synthesis and FPGA

```bash
# The RTL in hw/rtl/ can be synthesized with:
# - sattn_inst_decode.sv (instruction decoder)
# - rocc_sattn.sv (accelerator with CSR interface)
#
# Wire test program ROM and run on FPGA
```

## Common Issues

### Issue: Illegal Instruction

**Symptom**: Tests crash with "illegal instruction" error

**Solutions**:
1. Compile without `-DSATTN_USE_CSR_INSTRUCTIONS` for MMIO mode
2. Use simulator with custom extension support
3. Verify toolchain supports `.word` directive

### Issue: CSR Access Violation

**Symptom**: Tests trap on CSR read/write

**Solutions**:
1. Ensure CSRs are in custom range (0x7C0-0x7DF)
2. Run in machine mode (M-mode) if CSRs require privilege
3. Check simulator CSR support

### Issue: Instruction Encoding Mismatch

**Symptom**: Disassembly shows unexpected instruction encoding

**Solutions**:
1. Verify `SATTN_OPCODE` is 0x2B in sattn_isa.h
2. Check macro definitions in sattn_isa.h
3. Ensure no toolchain optimization of `.word` directives

## Continuous Integration

### GitHub Actions Example

```yaml
name: RISC-V Instruction Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install RISC-V Toolchain
        run: |
          wget https://github.com/riscv/riscv-gnu-toolchain/releases/...
          tar xzf riscv64-...
      - name: Build Tests
        run: |
          cd backends/rocc/tests
          make all
      - name: Run Tests (MMIO mode)
        run: |
          cd backends/rocc/tests
          qemu-riscv64 ./test_instruction_encoding
```

## Contributing

When adding new tests:
1. Follow existing test structure (TEST_ASSERT macros)
2. Add both C and assembly versions if applicable
3. Update this README with new test descriptions
4. Ensure tests work in both MMIO and CSR modes

## References

- `hw/spec/instruction_encoding.md` - Instruction specification
- `hw/spec/csr_map.md` - CSR address map
- `hw/spec/assembly_guide.md` - Programming guide with examples
- `hw/spec/sattn_isa.h` - C header with macros

