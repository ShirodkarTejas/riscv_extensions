# RoCC Sparse Attention Accelerator (Skeleton)

This is a minimal SystemVerilog MMIO-style skeleton that models a RoCC-like accelerator for sparse attention primitives. It exposes a register file to latch command descriptors and returns done after a fixed latency. Replace the FSM with real datapaths (gather/DMA, spdot_bsr, softmax_fused, spmm_bsr).

## Files
- `rocc_sattn.sv`: MMIO register block + simple RUN/DONE FSM

## Register map (byte offsets)
- 0x0000: `q_base` (64b)
- 0x0008: `k_base` (64b)
- 0x0010: `v_base` (64b)
- 0x0018: `o_base` (64b)
- 0x0020: `idx_base` (64b)
- 0x0028: `stride_base` (64b)
- 0x0030: `m_rows` (32b)
- 0x0038: `head_dim_d` (32b)
- 0x0040: `block_size` (32b)
- 0x0048: `k_blocks` (32b)
- 0x0050: `s_tokens` (32b)
- 0x0058: `scale_fp_bits` (32b float bits)
- 0x0060: `cmd` write: issue; read: status bits `[1]=busy, [0]=done`

## Command IDs (tentative)
- 0x10 blk_reduce
- 0x11 topk_idx
- 0x12 gath2d
- 0x13 scat2d
- 0x14 spdot_bsr
- 0x15 softmax_fused
- 0x16 spmm_bsr

## Integration notes
- Wrap this module behind a RoCC adapter or an AXI4-Lite bridge for MMIO.
- Replace the fixed-latency RUN state with actual pipelines connected to memory.
- Use `hw/spec/rocc_intrinsics.h` to define driver-side descriptors.
