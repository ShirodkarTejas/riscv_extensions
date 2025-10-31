// Simple MMIO-style skeleton for a RoCC-like Sparse Attention accelerator
// This does not implement compute; it latches command descriptors and asserts done.

module rocc_sattn #(
  parameter ADDR_WIDTH = 16,
  parameter DATA_WIDTH = 64
)(
  input  logic                 clk,
  input  logic                 rstn,

  // MMIO interface (simple register file)
  input  logic                 mmio_wen,
  input  logic                 mmio_ren,
  input  logic [ADDR_WIDTH-1:0] mmio_addr,
  input  logic [DATA_WIDTH-1:0] mmio_wdata,
  output logic [DATA_WIDTH-1:0] mmio_rdata,

  // Status
  output logic                 busy,
  output logic                 done
);

  // Register map (offsets in 8-byte words)
  localparam REG_Q_BASE    = 16'h0000;
  localparam REG_K_BASE    = 16'h0008;
  localparam REG_V_BASE    = 16'h0010;
  localparam REG_O_BASE    = 16'h0018;
  localparam REG_IDX_BASE  = 16'h0020;
  localparam REG_STRD_BASE = 16'h0028;
  localparam REG_M_ROWS    = 16'h0030;
  localparam REG_HEAD_D    = 16'h0038;
  localparam REG_BLOCK_SZ  = 16'h0040;
  localparam REG_K_BLOCKS  = 16'h0048;
  localparam REG_S_TOKENS  = 16'h0050;
  localparam REG_SCALE_FP  = 16'h0058;
  localparam REG_CMD       = 16'h0060; // write to issue; read status
  localparam REG_ACC_SUM   = 16'h0068; // read checksum from core
  localparam REG_SOF_SUM   = 16'h0080; // read softmax_fused checksum
  localparam REG_SPM_SUM   = 16'h0088; // read spmm_bsr checksum
  localparam REG_G_CYCLES  = 16'h0090; // gather cycles
  localparam REG_M_CYCLES  = 16'h0098; // mac cycles (spdot run)
  localparam REG_DMA_BYTES = 16'h00A0; // dma-like bytes (q/k writes)
  localparam REG_DMA_QBYTES= 16'h00A8; // dma bytes into Q spad
  localparam REG_DMA_KBYTES= 16'h00B0; // dma bytes into K spad
  localparam REG_GQA_GSZ  = 16'h00C8; // grouped-query heads per KV group
  localparam REG_COMP_BS  = 16'h00D0; // compression block size (0 disables)
  localparam REG_IDX_WADDR = 16'h0070; // write index RAM address
  localparam REG_IDX_WDATA = 16'h0078; // write index RAM data (commit)

  logic [63:0] q_base, k_base, v_base, o_base, idx_base, strd_base;
  logic [31:0] m_rows, head_dim_d, block_size, k_blocks, s_tokens;
  logic [31:0] scale_fp_bits;
  logic [31:0] gqa_group_size;
  logic [31:0] comp_block_size;

  typedef enum logic [7:0] {
    CMD_NOP         = 8'h00,
    CMD_BLK_REDUCE  = 8'h10,
    CMD_TOPK_IDX    = 8'h11,
    CMD_GATH2D      = 8'h12,
    CMD_SCAT2D      = 8'h13,
    CMD_SPDOT_BSR   = 8'h14,
    CMD_SOFTMAX_FUS = 8'h15,
    CMD_SPMM_BSR    = 8'h16
  } cmd_e;

  logic [7:0] cmd_reg;
  logic       start_pulse;
  assign start_pulse = 1'b0;
  logic [63:0] acc_sum;
  logic [63:0] gather_cycles, mac_cycles, dma_bytes;
  logic [63:0] dma_q_bytes, dma_k_bytes;

  // Simple ready/done FSM
  typedef enum logic [1:0] {IDLE, GATHER, RUN, DONE} state_e;
  state_e state, state_n;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  // spdot_bsr core integration
  logic core_start, core_busy, core_done;
  wire  is_spdot = (cmd_reg == CMD_SPDOT_BSR);

  spdot_bsr_core u_spdot (
    .clk(clk), .rstn(rstn),
    .start(core_start),
    .m_rows(m_rows[15:0]), .head_dim_d(head_dim_d[15:0]), .s_tokens(s_tokens[15:0]), .block_size(block_size[15:0]),
    .q_raddr(q_raddr), .q_rdata(q_rdata),
    .k_raddr(k_raddr), .k_rdata(k_rdata),
    .idx_rd_addr(core_idx_addr), .idx_rd_data(idx_rd_data),
    .busy(core_busy), .done(core_done), .checksum_out(core_sum)
  );

  wire [63:0] core_sum;
  wire [63:0] sof_sum;
  wire [63:0] spm_sum;
  /* verilator lint_off UNUSED */
  logic unused_tie;
  assign unused_tie = mmio_ren ^ start_pulse ^ core_busy ^ g_busy ^ sof_busy ^ spm_busy;
  /* verilator lint_on UNUSED */

  // gather stub
  logic g_start, g_busy, g_done;
  logic        q_wen;
  logic [15:0] q_waddr;
  logic [31:0] q_wdata;
  logic        k_wen;
  logic [15:0] k_waddr;
  logic [31:0] k_wdata;
  logic [15:0] q_raddr, k_raddr;
  logic [31:0] q_rdata, k_rdata;
  // index RAM interface
  logic [15:0] idx_rd_addr;
  logic [15:0] core_idx_addr;
  logic [15:0] idx_rd_data;
  gather2d_stub u_gather (
    .clk(clk), .rstn(rstn), .start(g_start), .s_tokens(s_tokens[15:0]), .head_dim_d(head_dim_d[15:0]), .block_size(block_size[15:0]),
    .stride_d(16'd1), .stride_t(16'd1),
    .q_wen(q_wen), .q_waddr(q_waddr), .q_wdata(q_wdata),
    .k_wen(k_wen), .k_waddr(k_waddr), .k_wdata(k_wdata),
    .idx_rd_addr(idx_rd_addr), .idx_rd_data(idx_rd_data),
    .busy(g_busy), .done(g_done)
  );

  // Q/K scratchpads
  spad #(.ADDR_WIDTH(16), .DATA_WIDTH(32)) u_q_spad (
    .clk(clk), .wen(q_wen), .waddr(q_waddr), .wdata(q_wdata), .raddr(q_raddr), .rdata(q_rdata)
  );
  spad #(.ADDR_WIDTH(16), .DATA_WIDTH(32)) u_k_spad (
    .clk(clk), .wen(k_wen), .waddr(k_waddr), .wdata(k_wdata), .raddr(k_raddr), .rdata(k_rdata)
  );

  // index RAM (depth 64K entries)
  logic        idx_wen;
  logic [15:0] idx_waddr;
  logic [15:0] idx_wdata;
  // Mux read port between gather and core
  logic [15:0] idx_raddr_mux;
  assign idx_raddr_mux = (state == GATHER) ? idx_rd_addr : core_idx_addr;
  idx_ram #(.ADDR_WIDTH(16), .DATA_WIDTH(16)) u_idx (
    .clk(clk), .rstn(rstn), .wen(idx_wen), .waddr(idx_waddr), .wdata(idx_wdata),
    .raddr(idx_raddr_mux), .rdata(idx_rd_data)
  );

  // softmax fused stub
  logic sof_start, sof_busy, sof_done;
  softmax_fused_stub u_soft (
    .clk(clk), .rstn(rstn), .start(sof_start), .m_rows(m_rows[15:0]), .s_tokens(s_tokens[15:0]),
    .busy(sof_busy), .done(sof_done), .checksum_out(sof_sum)
  );

  // spmm bsr stub
  logic spm_start, spm_busy, spm_done;
  spmm_bsr_stub u_spmm (
    .clk(clk), .rstn(rstn), .start(spm_start), .m_rows(m_rows[15:0]), .s_tokens(s_tokens[15:0]), .head_dim_d(head_dim_d[15:0]),
    .busy(spm_busy), .done(spm_done), .checksum_out(spm_sum)
  );

  // MMIO write
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      q_base <= '0; k_base <= '0; v_base <= '0; o_base <= '0;
      idx_base <= '0; strd_base <= '0;
      m_rows <= '0; head_dim_d <= '0; block_size <= '0; k_blocks <= '0; s_tokens <= '0;
      scale_fp_bits <= '0; cmd_reg <= '0; idx_wen <= 1'b0; idx_waddr <= '0; idx_wdata <= '0;
      dma_q_bytes <= 64'd0; dma_k_bytes <= 64'd0;
      gqa_group_size <= 32'd1; comp_block_size <= 32'd0;
    end else if (mmio_wen) begin
      unique case (mmio_addr)
        REG_Q_BASE:    q_base <= mmio_wdata;
        REG_K_BASE:    k_base <= mmio_wdata;
        REG_V_BASE:    v_base <= mmio_wdata;
        REG_O_BASE:    o_base <= mmio_wdata;
        REG_IDX_BASE:  idx_base <= mmio_wdata;
        REG_STRD_BASE: strd_base <= mmio_wdata;
        REG_M_ROWS:    m_rows <= mmio_wdata[31:0];
        REG_HEAD_D:    head_dim_d <= mmio_wdata[31:0];
        REG_BLOCK_SZ:  block_size <= mmio_wdata[31:0];
        REG_K_BLOCKS:  k_blocks <= mmio_wdata[31:0];
        REG_S_TOKENS:  s_tokens <= mmio_wdata[31:0];
        REG_SCALE_FP:  scale_fp_bits <= mmio_wdata[31:0];
        REG_GQA_GSZ:   gqa_group_size <= mmio_wdata[31:0];
        REG_COMP_BS:   comp_block_size <= mmio_wdata[31:0];
        REG_CMD:       cmd_reg <= mmio_wdata[7:0];
        REG_IDX_WADDR: begin idx_waddr <= mmio_wdata[15:0]; idx_wen <= 1'b0; end
        REG_IDX_WDATA: begin idx_wdata <= mmio_wdata[15:0]; idx_wen <= 1'b1; end
        default: ;
      endcase
    end else begin
      idx_wen <= 1'b0;
    end
  end

  // MMIO read
  always_comb begin
    mmio_rdata = '0;
    unique case (mmio_addr)
      REG_Q_BASE:    mmio_rdata = q_base;
      REG_K_BASE:    mmio_rdata = k_base;
      REG_V_BASE:    mmio_rdata = v_base;
      REG_O_BASE:    mmio_rdata = o_base;
      REG_IDX_BASE:  mmio_rdata = idx_base;
      REG_STRD_BASE: mmio_rdata = strd_base;
      REG_M_ROWS:    mmio_rdata = {32'd0, m_rows};
      REG_HEAD_D:    mmio_rdata = {32'd0, head_dim_d};
      REG_BLOCK_SZ:  mmio_rdata = {32'd0, block_size};
      REG_K_BLOCKS:  mmio_rdata = {32'd0, k_blocks};
      REG_S_TOKENS:  mmio_rdata = {32'd0, s_tokens};
      REG_SCALE_FP:  mmio_rdata = {32'd0, scale_fp_bits};
      REG_GQA_GSZ:   mmio_rdata = {32'd0, gqa_group_size};
      REG_COMP_BS:   mmio_rdata = {32'd0, comp_block_size};
      REG_CMD:       mmio_rdata = {62'd0, (state==RUN), (state==DONE)}; // [1]=busy, [0]=done
      REG_ACC_SUM:   mmio_rdata = acc_sum;
      REG_SOF_SUM:   mmio_rdata = sof_sum;
      REG_SPM_SUM:   mmio_rdata = spm_sum;
      REG_G_CYCLES:  mmio_rdata = gather_cycles;
      REG_M_CYCLES:  mmio_rdata = mac_cycles;
      REG_DMA_BYTES: mmio_rdata = dma_bytes;
      REG_DMA_QBYTES:mmio_rdata = dma_q_bytes;
      REG_DMA_KBYTES:mmio_rdata = dma_k_bytes;
      default:       mmio_rdata = '0;
    endcase
  end

  // FSM: when cmd written, go RUN; for spdot use core_done, otherwise variable latency
  logic cmd_seen;
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      state <= IDLE; cmd_seen <= 1'b0;
    end else begin
      state <= state_n;
      if (mmio_wen && mmio_addr == REG_CMD) cmd_seen <= 1'b1;
      else if (state == DONE) cmd_seen <= 1'b0;
    end
  end

  logic [15:0] run_cnt;
  logic [15:0] run_len;

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      run_cnt <= '0; run_len <= 16'd16;
      gather_cycles <= 64'd0; mac_cycles <= 64'd0; dma_bytes <= 64'd0;
    end else begin
      // Hoisted temporaries for Verilator friendliness
      logic [15:0] base_len;
      logic [31:0] scale_mul;
      logic [15:0] len16;
      /* verilator lint_off UNUSEDSIGNAL */
      logic [31:0] mul32;
      /* verilator lint_on UNUSEDSIGNAL */
      if (state == IDLE && cmd_seen && cmd_reg != CMD_NOP) begin
        // crude latency model: function of m_rows, head_dim, s_tokens (widen to 16b)
        base_len = (m_rows[15:0] + head_dim_d[15:0] + s_tokens[15:0]);
        if (base_len == 16'd0) base_len = 16'd16;
        // scale by GQA group size and adjust for compression block size vs selection block size
        scale_mul = (gqa_group_size == 0) ? 32'd1 : gqa_group_size;
        if (comp_block_size != 32'd0 && block_size != 32'd0) begin
          // fewer effective tokens when compression blocks are smaller than selection blocks
          scale_mul = (scale_mul * comp_block_size) / block_size;
          if (scale_mul == 32'd0) scale_mul = 32'd1;
        end
        mul32 = (32'(base_len) * scale_mul);
        len16 = mul32[15:0];
        run_len <= (len16 != 16'd0) ? len16 : 16'd16;
      end
      if (state == RUN) run_cnt <= run_cnt + 16'd1; else run_cnt <= '0;
      // Latch core checksum after entering DONE to allow child to register its output
      if (state == DONE && is_spdot) acc_sum <= core_sum;
      // Per-cycle counters
      if (state == GATHER) begin
        gather_cycles <= gather_cycles + 64'd1;
      end
      if (state == RUN && is_spdot) begin
        mac_cycles <= mac_cycles + 64'd1;
      end
      // DMA bytes count writes into Q/K scratchpads (handle simultaneous writes)
      begin
        logic [63:0] dma_inc;
        dma_inc = 64'd0;
        if (q_wen) begin dma_inc = dma_inc + 64'd4; dma_q_bytes <= dma_q_bytes + 64'd4; end
        if (k_wen) begin dma_inc = dma_inc + 64'd4; dma_k_bytes <= dma_k_bytes + 64'd4; end
        if (dma_inc != 64'd0) dma_bytes <= dma_bytes + dma_inc;
      end
      // Remove legacy transition-based bumps; counters now reflect per-cycle and per-write events only
    end
  end

  always_comb begin
    state_n = state;
    core_start = 1'b0;
    g_start = 1'b0;
    sof_start = 1'b0;
    spm_start = 1'b0;
    unique case (state)
      IDLE: if (cmd_seen && cmd_reg != CMD_NOP) begin
               if (is_spdot) begin
                 state_n = GATHER;
                 g_start = 1'b1;
               end else if (cmd_reg == CMD_SOFTMAX_FUS) begin
                  state_n = RUN;
                  sof_start = 1'b1;
               end else if (cmd_reg == CMD_SPMM_BSR) begin
                   state_n = RUN;
                   spm_start = 1'b1;
               end else begin
                 state_n = RUN;
               end
             end
      GATHER: begin
               if (g_done) begin
                 state_n = RUN;
                 core_start = 1'b1;
               end
              end
      RUN:   begin
              if (is_spdot) begin
                if (core_done) state_n = DONE;
              end else if (cmd_reg == CMD_SOFTMAX_FUS) begin
                 if (sof_done) state_n = DONE;
              end else if (cmd_reg == CMD_SPMM_BSR) begin
                  if (spm_done) state_n = DONE;
               end else begin
                 if (run_cnt >= run_len) state_n = DONE; // variable latency placeholder
               end
             end
      DONE: state_n = IDLE;
      default: state_n = IDLE;
    endcase
  end

endmodule


