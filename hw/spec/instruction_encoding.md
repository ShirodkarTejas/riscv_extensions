# RISC-V Custom Instruction Encoding for Sparse Attention

This document specifies the custom RISC-V instruction encoding for the Sparse Attention accelerator primitives. The design follows RISC-V custom extension guidelines and provides a foundation for future standardization.

## Opcode Selection

We use the **`custom-1`** opcode space allocated by the RISC-V specification for user-defined custom extensions:

- **Opcode**: `0x2B` (7 bits: `0101011`)
- **Opcode Space**: `custom-1` from RISC-V specification (reserved for custom extensions)

Alternative: `custom-0` (`0x0B`, `0001011`) can be used if `custom-1` conflicts with other extensions.

## Instruction Format

All sparse attention instructions use an **R-type format variant**:

```
 31          25 24     20 19     15 14   12 11      7 6        0
┌──────────────┬─────────┬─────────┬───────┬─────────┬──────────┐
│    funct7    │   rs2   │   rs1   │ funct3│   rd    │  opcode  │
│   [31:25]    │ [24:20] │ [19:15] │[14:12]│ [11:7]  │  [6:0]   │
└──────────────┴─────────┴─────────┴───────┴─────────┴──────────┘
    7 bits       5 bits    5 bits   3 bits   5 bits    7 bits
```

### Field Definitions

- **opcode [6:0]**: `0x2B` (`0101011`) - custom-1 opcode
- **rd [11:7]**: Destination register (typically `x0` for status-only ops, or target register for result)
- **funct3 [14:12]**: Primary operation selector (primitive type)
- **rs1 [19:15]**: Source register 1 (mode/flags or descriptor pointer for legacy compatibility)
- **rs2 [24:20]**: Source register 2 (mode/flags or additional parameters)
- **funct7 [31:25]**: Secondary function code (currently `0x00`, reserved for future variants)

## Primitive Encoding (funct3 Field)

The `funct3` field selects which sparse attention primitive to execute:

| funct3 | Binary | Primitive        | Description                                    |
|--------|--------|------------------|------------------------------------------------|
| 0      | 000    | blk_reduce       | Block reduction (mean/max) over K blocks       |
| 1      | 001    | topk_idx         | Top-k block index selection                    |
| 2      | 010    | gath2d           | 2D block-structured gather                     |
| 3      | 011    | scat2d           | 2D block-structured scatter                    |
| 4      | 100    | spdot_bsr        | Sparse block dot product (Q × K^T)             |
| 5      | 101    | softmax_fused    | Fused softmax (scale, exp, normalize)          |
| 6      | 110    | spmm_bsr         | Sparse block matrix multiply (Attention × V)   |
| 7      | 111    | (reserved)       | Reserved for NOP or future extensions          |

## Parameter Passing Model

All primitives use **CSR-based parameter staging**. Before issuing an instruction, software must:

1. Write parameters to custom CSRs (see `csr_map.md`)
2. Issue the instruction via inline assembly or intrinsic
3. Poll status CSR or wait for completion

### Register Usage

- **rs1**: 
  - Mode/flags for simple operations
  - Descriptor pointer for legacy MMIO compatibility mode
  - Can be `x0` if not needed
  
- **rs2**: 
  - Additional mode bits (reduction type, sort flags, etc.)
  - Can be `x0` if not needed
  
- **rd**: 
  - Status/result register (0=success, non-zero=error code)
  - Use `x0` if status not needed
  - Some operations may return scalar results here

## Assembly Syntax

Proposed assembly mnemonics (to be supported by custom assembler or macros):

```assembly
sattn.blk_reduce   rd, rs1, rs2    # Block reduction
sattn.topk_idx     rd, rs1, rs2    # Top-K index selection
sattn.gath2d       rd, rs1, rs2    # 2D gather
sattn.scat2d       rd, rs1, rs2    # 2D scatter
sattn.spdot_bsr    rd, rs1, rs2    # Sparse dot product
sattn.softmax_fused rd, rs1, rs2   # Fused softmax
sattn.spmm_bsr     rd, rs1, rs2    # Sparse matrix multiply
```

Example (conceptual):
```assembly
# Setup parameters via CSRs
li   t0, 128              # m_rows
csrw 0x7C6, t0
li   t0, 64               # head_dim_d
csrw 0x7C7, t0
# ... (more CSR writes)

# Issue sparse dot product
sattn.spdot_bsr x0, x0, x0   # All params from CSRs, no status return
```

## Encoding Table

Complete encoding for all primitives (with funct7=0x00, rs1=x0, rs2=x0, rd=x0):

| Instruction         | Encoding (hex) | Binary                                  |
|---------------------|----------------|-----------------------------------------|
| sattn.blk_reduce    | 0x0000002B     | 0000000 00000 00000 000 00000 0101011  |
| sattn.topk_idx      | 0x0000102B     | 0000000 00000 00000 001 00000 0101011  |
| sattn.gath2d        | 0x0000202B     | 0000000 00000 00000 010 00000 0101011  |
| sattn.scat2d        | 0x0000302B     | 0000000 00000 00000 011 00000 0101011  |
| sattn.spdot_bsr     | 0x0000402B     | 0000000 00000 00000 100 00000 0101011  |
| sattn.softmax_fused | 0x0000502B     | 0000000 00000 00000 101 00000 0101011  |
| sattn.spmm_bsr      | 0x0000602B     | 0000000 00000 00000 110 00000 0101011  |

## Execution Model

1. **Non-blocking**: Instructions return immediately and execute asynchronously
2. **Completion**: Software polls status CSR (0x7CE) for busy/done flags
3. **Atomicity**: Each instruction is atomic with respect to CSR state snapshot
4. **Exceptions**: Illegal parameter configurations may raise exceptions or set error flags

## Future Extensions

The `funct7` field provides 128 encoding variants per primitive, enabling:
- Different data types (bf16, fp16, int8, int4)
- Algorithmic variants (different reduction modes, sparse formats)
- Precision/performance tradeoffs

Possible funct7 usage:
- `0x00`: Default (bf16 precision)
- `0x01`: FP16 precision
- `0x02`: INT8 quantized
- `0x03-0x7F`: Reserved

## Compatibility Notes

- **RV32/RV64**: Encoding is identical; CSR width adapts (see `csr_map.md`)
- **Privileged Levels**: Instructions are user-mode accessible
- **ISA Dependencies**: Base integer ISA (RV32I/RV64I) required; no dependency on F/D/V extensions
- **Conflicts**: Ensure no conflict with other custom extensions in your platform

## Validation

Hardware decoders should:
1. Match opcode `0x2B`
2. Decode funct3 to select primitive
3. Validate funct7 (reject unsupported variants)
4. Check CSR state validity before execution
5. Signal illegal instruction exception for reserved encodings (funct3=0x7 with non-zero funct7)

## Migration Path

This encoding is designed for evolution:

- **Phase 1** (current): Custom extension with CSR staging
- **Phase 2**: Formal ISA extension proposal (allocate official opcode)
- **Phase 3**: LLVM/GCC toolchain integration
- **Phase 4**: RISC-V International ratification (if suitable for standardization)

## References

- RISC-V ISA Specification Volume I (Unprivileged)
- RISC-V ISA Specification Volume II (Privileged)
- Custom extension guidelines: https://github.com/riscv/riscv-isa-manual

