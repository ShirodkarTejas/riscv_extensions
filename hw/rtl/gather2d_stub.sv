// gather2d stub: emulates block-structured gather for s_tokens blocks.
// Consumes a fixed number of cycles proportional to s_tokens; no memory accessed.

module gather2d_stub (
  input  logic        clk,
  input  logic        rstn,
  input  logic        start,
  input  logic [15:0] s_tokens,
  input  logic [15:0] head_dim_d,
  // write ports to core scratchpads
  output logic        q_wen,
  output logic [15:0] q_waddr,
  output logic [31:0] q_wdata,
  output logic        k_wen,
  output logic [15:0] k_waddr,
  output logic [31:0] k_wdata,
  // index RAM read interface
  output logic [15:0] idx_rd_addr,
  input  logic [15:0] idx_rd_data,
  output logic        busy,
  output logic        done
);

  typedef enum logic [1:0] {IDLE, RUN, DONE} state_e;
  state_e state, state_n;

  logic [15:0] cnt;
  logic [15:0] waddr;
  logic [15:0] tok_cnt;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) state <= IDLE; else state <= state_n;
  end

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      cnt <= '0; waddr <= '0; tok_cnt <= '0;
    end else if (state == RUN) begin
      cnt <= cnt + 16'd1;
      if (waddr + 1 < head_dim_d) begin
        waddr <= waddr + 16'd1;
      end else begin
        waddr <= 16'd0;
        if (tok_cnt + 1 < s_tokens) tok_cnt <= tok_cnt + 16'd1;
      end
    end else begin
      cnt <= '0; waddr <= '0; tok_cnt <= '0;
    end
  end

  always_comb begin
    state_n = state;
    // default no writes
    q_wen = 1'b0; q_waddr = '0; q_wdata = '0;
    k_wen = 1'b0; k_waddr = '0; k_wdata = '0;
    idx_rd_addr = tok_cnt;
    unique case (state)
      IDLE: if (start) state_n = RUN;
      RUN:  begin
              // emit one write per cycle to both Q and K buffers at address waddr
              q_wen = 1'b1; q_waddr = waddr; q_wdata = {idx_rd_data, waddr};
              k_wen = 1'b1; k_waddr = waddr; k_wdata = {idx_rd_data ^ 16'h0f0f, waddr};
              if (cnt >= s_tokens) state_n = DONE;
            end
      DONE: state_n = IDLE;
      default: state_n = IDLE;
    endcase
  end

endmodule


