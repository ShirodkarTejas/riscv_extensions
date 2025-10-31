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
  localparam REG_IDX_WADDR = 16'h0070; // write index RAM address
  localparam REG_IDX_WDATA = 16'h0078; // write index RAM data (commit)

  logic [63:0] q_base, k_base, v_base, o_base, idx_base, strd_base;
  logic [31:0] m_rows, head_dim_d, block_size, k_blocks, s_tokens;
  logic [31:0] scale_fp_bits;

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
  logic [63:0] acc_sum;

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
    .m_rows(m_rows[15:0]), .head_dim_d(head_dim_d[15:0]), .s_tokens(s_tokens[15:0]),
    .q_raddr(q_raddr), .q_rdata(q_rdata),
    .k_raddr(k_raddr), .k_rdata(k_rdata),
    .busy(core_busy), .done(core_done), .checksum_out(core_sum)
  );

  logic [63:0] core_sum;

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
  logic [15:0] idx_rd_data;
  gather2d_stub u_gather (
    .clk(clk), .rstn(rstn), .start(g_start), .s_tokens(s_tokens[15:0]), .head_dim_d(head_dim_d[15:0]),
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
  idx_ram #(.ADDR_WIDTH(16), .DATA_WIDTH(16)) u_idx (
    .clk(clk), .rstn(rstn), .wen(idx_wen), .waddr(idx_waddr), .wdata(idx_wdata),
    .raddr(idx_rd_addr), .rdata(idx_rd_data)
  );

  // MMIO write
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      q_base <= '0; k_base <= '0; v_base <= '0; o_base <= '0;
      idx_base <= '0; strd_base <= '0;
      m_rows <= '0; head_dim_d <= '0; block_size <= '0; k_blocks <= '0; s_tokens <= '0;
      scale_fp_bits <= '0; cmd_reg <= '0; idx_wen <= 1'b0; idx_waddr <= '0; idx_wdata <= '0;
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
      REG_CMD:       mmio_rdata = {62'd0, (state==RUN), (state==DONE)}; // [1]=busy, [0]=done
      REG_ACC_SUM:   mmio_rdata = acc_sum;
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
    end else begin
      if (state == IDLE && cmd_seen && cmd_reg != CMD_NOP) begin
        // crude latency model: function of m_rows, head_dim, s_tokens
        run_len <= (m_rows[7:0] + head_dim_d[7:0] + s_tokens[7:0]);
        if (run_len == 16'd0) run_len <= 16'd16;
      end
      if (state == RUN) run_cnt <= run_cnt + 16'd1; else run_cnt <= '0;
      if (state == RUN && is_spdot && core_done) acc_sum <= core_sum;
    end
  end

  always_comb begin
    state_n = state;
    core_start = 1'b0;
    g_start = 1'b0;
    unique case (state)
      IDLE: if (cmd_seen && cmd_reg != CMD_NOP) begin
               if (is_spdot) begin
                 state_n = GATHER;
                 g_start = 1'b1;
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
               end else begin
                 if (run_cnt >= run_len) state_n = DONE; // variable latency placeholder
               end
             end
      DONE: state_n = IDLE;
      default: state_n = IDLE;
    endcase
  end

endmodule


