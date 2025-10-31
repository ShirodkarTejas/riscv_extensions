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

  // Simple ready/done FSM
  typedef enum logic [1:0] {IDLE, RUN, DONE} state_e;
  state_e state, state_n;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  // MMIO write
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      q_base <= '0; k_base <= '0; v_base <= '0; o_base <= '0;
      idx_base <= '0; strd_base <= '0;
      m_rows <= '0; head_dim_d <= '0; block_size <= '0; k_blocks <= '0; s_tokens <= '0;
      scale_fp_bits <= '0; cmd_reg <= '0;
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
        default: ;
      endcase
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
      default:       mmio_rdata = '0;
    endcase
  end

  // FSM: when cmd written, go RUN for a few cycles, then DONE
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

  logic [3:0] run_cnt;

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) run_cnt <= '0; else if (state == RUN) run_cnt <= run_cnt + 4'd1; else run_cnt <= '0;
  end

  always_comb begin
    state_n = state;
    unique case (state)
      IDLE: if (cmd_seen && cmd_reg != CMD_NOP) state_n = RUN;
      RUN:  if (run_cnt == 4'd8) state_n = DONE; // fixed latency placeholder
      DONE: state_n = IDLE;
      default: state_n = IDLE;
    endcase
  end

endmodule


