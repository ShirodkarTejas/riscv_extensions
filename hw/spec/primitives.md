# Sparse Attention Custom Primitives (RoCC Route A)

This spec defines a minimal set of RoCC-accessible primitives to accelerate sparse attention on RISC-V. The goal is to offload the irregular parts (selection, gather/scatter) and the softmax bottleneck while keeping a simple programmer’s model.

## Primitives

- blk_reduce
  - Purpose: per-block reduction (mean/max) over K tiles to produce block scores for selection.
  - Inputs: K tile [block_size, D]; mode ∈ {mean, max};
  - Output: 1×D (mean) or 1 scalar (max), written to scratchpad.

- topk_idx
  - Purpose: select top-k blocks per query row from block scores.
  - Inputs: scores [num_blocks]; k; optional threshold;
  - Output: indices [k] (compact, sorted/unsorted by flag).

- gath2d / scat2d
  - Purpose: block-structured gather/scatter with stride tables.
  - Inputs: base ptr, block_size, D, index table [k]; stride table;
  - Output: compacted tensor [k*block_size, D] to/from scratchpad.

- spdot_bsr
  - Purpose: block-sparse dot Q × Kᵀ for a BSR pattern.
  - Inputs: Q tile [M, D]; K blocks [k, block_size, D]; block index table;
  - Output: score matrix [M, k*block_size].

- softmax_fused
  - Purpose: scale + max-reduce + exp + sum-reduce + normalize (tile-local).
  - Inputs: scores [M, S]; scale; epsilon;
  - Output: probabilities [M, S].

- spmm_bsr
  - Purpose: block-sparse AV multiply.
  - Inputs: P [M, S]; V blocks [k, block_size, D]; block index table;
  - Output: O [M, D].

Optional (future): gelu_ln_fuse for attention-FFN fusion.

## Programmer’s model

- Tiling: M (rows per tile), D (head dim), block_size (tokens per block), S=k*block_size (selected tokens).
- Scratchpad: capacity must hold {Q[M,D], K[k,block_size,D], V[k,block_size,D], scores[M,S], probs[M,S]}.
- Alignment: 64B alignment for all base pointers; D multiple of 16; block_size multiple of 16.
- Throughput/latency (targets):
  - spdot_bsr: peak-close MMA-style throughput; hide gather via prefetch FIFO.
  - softmax_fused: tree reductions within tile; 1 cycle per exp in CORDIC/LUT pipeline.

## RoCC command IDs and Custom Instruction Encoding

### Legacy MMIO Command IDs
- 0x10: blk_reduce
- 0x11: topk_idx
- 0x12: gath2d
- 0x13: scat2d
- 0x14: spdot_bsr
- 0x15: softmax_fused
- 0x16: spmm_bsr

### Custom RISC-V Instruction Encoding

The primitives are also exposed as custom RISC-V instructions using the `custom-1` opcode (0x2B):

**Instruction Format**: R-type
```
[funct7=0x00][rs2][rs1][funct3][rd][0x2B]
```

**funct3 Mapping** (primitive selector):
- 000 (0x0): blk_reduce
- 001 (0x1): topk_idx
- 010 (0x2): gath2d
- 011 (0x3): scat2d
- 100 (0x4): spdot_bsr
- 101 (0x5): softmax_fused
- 110 (0x6): spmm_bsr
- 111 (0x7): reserved

**Parameter Passing**: Via Custom CSRs (0x7C0-0x7DF)
- Base pointers: 0x7C0-0x7C5 (q, k, v, o, idx, stride)
- Dimensions: 0x7C6-0x7CB (m_rows, head_dim_d, block_size, k_blocks, s_tokens, scale)
- Status/control: 0x7CE-0x7CF

**Programming Models**:
1. **Instruction-based** (hardware): CSR writes + custom instruction issue
2. **MMIO-based** (simulation): Memory-mapped register writes

Both models are supported via a unified C API that selects the appropriate backend at compile time.

For details, see:
- `instruction_encoding.md` - Complete instruction specification
- `csr_map.md` - CSR address map and semantics
- `assembly_guide.md` - Programming examples and best practices

Completion is signaled via status CSR (0x7CE) DONE bit or MMIO status register.

## Operand semantics

- Base ptr CSRs: q_base, k_base, v_base, o_base, idx_base, stride_base.
- Size CSRs: m_rows, head_dim_d, block_size, k_blocks, s_tokens, scale_fp.
- Strides in elements.

## Micro-architecture sketch

- Scratchpad SRAM partitioned into banks for Q/K/V/score/prob tiles.
- DMA engine for block-gather/scatter with 2D stride support.
- Index FIFO to stream selected block IDs into gather DMA.
- MAC array organized as [D×M] × [D×S] outer-product tiles.
- Reduction tree for max/sum; exp implemented via LUT or polynomial.

## Analytical cycle estimates (high-level)

Let: M, D, S=k*block_size; BW = SRAM bandwidth (bytes/cycle); TMAC = MACs/cycle.

- spdot_bsr cycles ≈ ceil((M*D*S)/TMAC) + gather_cost
- softmax_fused cycles ≈ M*(log2(S) + S/exp_throughput)
- spmm_bsr cycles ≈ ceil((M*S*D)/TMAC)
- gather_cost ≈ bytes_moved/BW with overlap factor α∈[0,1]

See `cycle_model.py` for parameterized estimates.
