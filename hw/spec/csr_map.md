# Custom CSR Map for Sparse Attention Accelerator

This document defines the Control and Status Register (CSR) allocation for the sparse attention custom instructions. These CSRs hold parameters, control state, and status information for the accelerator primitives.

## CSR Address Space

We use the custom CSR range allocated by RISC-V for non-standard extensions:

- **Base Address**: `0x7C0` (1984 decimal)
- **Range**: `0x7C0` - `0x7DF` (32 CSRs)
- **Privilege Level**: User-mode accessible (URW - User Read/Write)
- **Address Space**: Custom/non-standard (per RISC-V spec section on custom CSRs)

**Note**: These CSRs are in the "reserved for custom use" region and will not conflict with standard RISC-V CSRs.

## CSR Width

- **RV32**: All CSRs are 32 bits wide
- **RV64**: CSRs are 64 bits wide (or paired for 64-bit values on RV32)

For 64-bit pointer values on RV32, we use consecutive CSR pairs (low word, then high word).

## CSR Allocation Table

### Base Pointers (0x7C0 - 0x7C5)

64-bit pointers to memory buffers. On RV32, each uses two consecutive CSRs.

| Address | Name         | Width  | Description                                      |
|---------|--------------|--------|--------------------------------------------------|
| 0x7C0   | SATTN_Q_BASE | 64-bit | Base pointer to Q (query) tensor                 |
| 0x7C1   | SATTN_K_BASE | 64-bit | Base pointer to K (key) tensor                   |
| 0x7C2   | SATTN_V_BASE | 64-bit | Base pointer to V (value) tensor                 |
| 0x7C3   | SATTN_O_BASE | 64-bit | Base pointer to O (output) tensor                |
| 0x7C4   | SATTN_IDX_BASE | 64-bit | Base pointer to index table (sparse patterns) |
| 0x7C5   | SATTN_STRIDE_BASE | 64-bit | Base pointer to stride table             |

**RV32 Pairing**: 
- 0x7C0 = Q_BASE[31:0], implied next CSR for Q_BASE[63:32]
- Hardware treats consecutive reads/writes as 64-bit pairs

**RV64**: Direct 64-bit access

### Dimension Parameters (0x7C6 - 0x7CB)

32-bit unsigned integers defining tensor shapes and tile sizes.

| Address | Name              | Width  | Description                              |
|---------|-------------------|--------|------------------------------------------|
| 0x7C6   | SATTN_M_ROWS      | 32-bit | Number of query rows (M) in current tile |
| 0x7C7   | SATTN_HEAD_DIM_D  | 32-bit | Head dimension (D)                       |
| 0x7C8   | SATTN_BLOCK_SIZE  | 32-bit | Sparse block size (tokens per block)     |
| 0x7C9   | SATTN_K_BLOCKS    | 32-bit | Number of K/V blocks                     |
| 0x7CA   | SATTN_S_TOKENS    | 32-bit | Total selected tokens (S = k*block_size) |
| 0x7CB   | SATTN_SCALE_FP    | 32-bit | Attention scale factor (as float bits)   |

**Valid Ranges**:
- `M_ROWS`: 1-4096 (typically 16, 32, 64)
- `HEAD_DIM_D`: 16-256, must be multiple of 16 (typically 64, 128)
- `BLOCK_SIZE`: 16-256, must be multiple of 16 (typically 32, 64, 128)
- `K_BLOCKS`: 1-1024
- `S_TOKENS`: 1-32768 (typically <= 2048)
- `SCALE_FP`: IEEE 754 single-precision float bits (e.g., 1/sqrt(D))

### Extended Parameters (0x7CC - 0x7CD)

Additional parameters for advanced features.

| Address | Name                | Width  | Description                                 |
|---------|---------------------|--------|---------------------------------------------|
| 0x7CC   | SATTN_GQA_GROUP_SIZE| 32-bit | Grouped-query attention: heads per KV group |
| 0x7CD   | SATTN_COMP_BLOCK_SIZE| 32-bit| Compression block size (0=disabled)         |

**Usage**:
- `GQA_GROUP_SIZE`: Set to 1 for standard multi-head attention (MHA), >1 for GQA
- `COMP_BLOCK_SIZE`: Experimental - compression block size, typically 0 (disabled)

### Control and Status (0x7CE - 0x7CF)

| Address | Name           | Width  | Access | Description                           |
|---------|----------------|--------|--------|---------------------------------------|
| 0x7CE   | SATTN_STATUS   | 32-bit | R/W    | Status and control register           |
| 0x7CF   | SATTN_ERROR    | 32-bit | RO     | Error code register                   |

#### SATTN_STATUS (0x7CE) - Bit Fields

```
Bits [31:16]: Reserved (read as 0)
Bit  [15:8]:  Command ID (last issued command)
Bit  [7:4]:   Reserved
Bit  [3]:     ERROR - Operation error flag
Bit  [2]:     READY - Accelerator ready for new command
Bit  [1]:     BUSY - Operation in progress
Bit  [0]:     DONE - Operation completed
```

**Read Access**:
- Software polls BUSY (bit 1) and DONE (bit 0)
- Typical polling: `while (!(CSR_READ(SATTN_STATUS) & 0x1)) {}`

**Write Access**:
- Writing DONE=1 clears the done flag
- Writing READY=1 performs soft reset

#### SATTN_ERROR (0x7CF) - Error Codes

| Code | Name                    | Description                           |
|------|-------------------------|---------------------------------------|
| 0x00 | NO_ERROR                | No error                              |
| 0x01 | INVALID_DIMENSIONS      | Tensor dimensions out of range/invalid|
| 0x02 | MISALIGNED_POINTER      | Pointer not 64-byte aligned           |
| 0x03 | UNSUPPORTED_OPERATION   | funct7 variant not implemented        |
| 0x04 | SCRATCHPAD_OVERFLOW     | Tile size exceeds scratchpad capacity |
| 0x05 | TIMEOUT                 | Operation exceeded watchdog timer     |
| 0x06 | MEMORY_ERROR            | Memory access fault (DMA error)       |
| 0xFF | UNKNOWN_ERROR           | Unspecified error                     |

### Performance Counters (0x7D0 - 0x7D5)

Optional read-only CSRs for performance monitoring.

| Address | Name                | Width  | Description                        |
|---------|---------------------|--------|------------------------------------|
| 0x7D0   | SATTN_CYCLE_COUNT   | 64-bit | Total cycles for last operation    |
| 0x7D1   | SATTN_MAC_OPS       | 64-bit | MAC operations performed           |
| 0x7D2   | SATTN_GATHER_CYCLES | 64-bit | Cycles spent in gather/DMA         |
| 0x7D3   | SATTN_COMPUTE_CYCLES| 64-bit | Cycles spent in compute (MAC/exp)  |
| 0x7D4   | SATTN_DMA_BYTES     | 64-bit | Total bytes transferred via DMA    |
| 0x7D5   | SATTN_CACHE_HITS    | 64-bit | Scratchpad/cache hit count         |

**Note**: Performance counters are optional and may read as zero if not implemented.

### Hardware Capabilities (0x7D8 - 0x7D9)

Read-only CSRs identifying hardware features.

| Address | Name            | Width  | Description                              |
|---------|-----------------|--------|------------------------------------------|
| 0x7D8   | SATTN_HW_VERSION| 32-bit | Hardware version (major.minor.patch)     |
| 0x7D9   | SATTN_HW_CAPS   | 32-bit | Capability bitfield                      |

#### SATTN_HW_VERSION Format

```
Bits [31:24]: Major version
Bits [23:16]: Minor version
Bits [15:0]:  Patch version
```

Example: `0x01000000` = version 1.0.0

#### SATTN_HW_CAPS Bit Fields

```
Bit [0]: BSR_SUPPORT     - Block Sparse Row format support
Bit [1]: SLIDING_WINDOW  - Sliding window pattern support
Bit [2]: GQA_SUPPORT     - Grouped-Query Attention support
Bit [3]: COMPRESSION     - KV compression support
Bit [4]: INT8_QUANT      - INT8 quantization support
Bit [5]: INT4_QUANT      - INT4 quantization support
Bit [6]: BF16_PRECISION  - BF16 data type support
Bit [7]: FP16_PRECISION  - FP16 data type support
Bits [31:8]: Reserved
```

### Reserved (0x7DA - 0x7DF)

Reserved for future extensions.

| Address Range | Description                              |
|---------------|------------------------------------------|
| 0x7DA - 0x7DF | Reserved (read as 0, writes ignored)     |

## CSR Access Patterns

### Initialization Sequence

```c
// 1. Check hardware capabilities
uint32_t version = csr_read(SATTN_HW_VERSION);
uint32_t caps = csr_read(SATTN_HW_CAPS);

// 2. Setup base pointers
csr_write(SATTN_Q_BASE, (uint64_t)q_ptr);
csr_write(SATTN_K_BASE, (uint64_t)k_ptr);
csr_write(SATTN_V_BASE, (uint64_t)v_ptr);
csr_write(SATTN_O_BASE, (uint64_t)o_ptr);
csr_write(SATTN_IDX_BASE, (uint64_t)idx_ptr);
csr_write(SATTN_STRIDE_BASE, (uint64_t)stride_ptr);

// 3. Setup dimensions
csr_write(SATTN_M_ROWS, m_rows);
csr_write(SATTN_HEAD_DIM_D, head_dim);
csr_write(SATTN_BLOCK_SIZE, block_size);
csr_write(SATTN_K_BLOCKS, k_blocks);
csr_write(SATTN_S_TOKENS, s_tokens);
csr_write(SATTN_SCALE_FP, scale_bits);

// 4. Issue instruction
asm volatile("sattn.spdot_bsr x0, x0, x0");

// 5. Wait for completion
while (!(csr_read(SATTN_STATUS) & 0x1)) {}

// 6. Check for errors
uint32_t error = csr_read(SATTN_ERROR);
```

### RV32 64-bit Pointer Access

On RV32, 64-bit pointers require two CSR operations:

```c
// Write 64-bit pointer on RV32
uint64_t ptr = 0x123456789ABCDEF0ULL;
csr_write(SATTN_Q_BASE, (uint32_t)(ptr & 0xFFFFFFFF));        // Low word
csr_write(SATTN_Q_BASE + 1, (uint32_t)((ptr >> 32) & 0xFFFFFFFF)); // High word
```

Hardware implementations should handle this automatically, treating consecutive CSR pairs as a single 64-bit entity.

### Atomic Updates

For multi-threaded or multi-hart scenarios:
- CSRs are **not** automatically synchronized across harts
- Software must use standard RISC-V synchronization (fences, atomics) if sharing accelerator
- Recommended: One hart exclusively owns accelerator during operation

## Implementation Notes

### Hardware Requirements

1. **CSR Register File**: Implement 32 x 32-bit or 64-bit registers
2. **CSR Read/Write Logic**: Standard RISC-V CSR access via `csrrw`, `csrrs`, `csrrc`
3. **Snapshot Semantics**: When instruction issued, hardware snapshots all relevant CSRs atomically
4. **Status Updates**: Hardware updates SATTN_STATUS asynchronously during operation

### Software Requirements

1. **CSR Macros**: Use `<riscv_csr.h>` or custom macros for portable access
2. **Alignment**: Ensure all pointers are 64-byte aligned (hardware may check)
3. **Validation**: Check SATTN_ERROR after operations
4. **Ordering**: Use fence instructions if CSR writes must be visible to other harts

### Verification

Testbenches should verify:
- CSR read/write accessibility
- Proper reset values (0 for data CSRs, hardware ID for version/caps)
- Status flag transitions (READY → BUSY → DONE)
- Error code generation for invalid parameters

## Future Extensions

Additional CSRs may be added for:
- Multi-head configuration (batch processing)
- Power/performance hints (DVFS control)
- Debug and profiling features
- Security features (memory range checking)

## References

- RISC-V Privileged Architecture Specification (CSR numbering)
- Custom CSR guidelines
- `instruction_encoding.md` - Instruction format using these CSRs
- `sattn_isa.h` - C/Assembly macros for CSR access

