// spmm_bsr stub: consumes M rows by S tokens and D dims notionally
// Produces a checksum-like output. Placeholder for block-sparse AV multiply.

module spmm_bsr_stub (
  input  logic        clk,
  input  logic        rstn,
  input  logic        start,
  input  logic [15:0] m_rows,
  input  logic [15:0] s_tokens,
  input  logic [15:0] head_dim_d,
  output logic        busy,
  output logic        done,
  output logic [63:0] checksum_out
);

  typedef enum logic [1:0] {IDLE, RUN, DONE} state_e;
  state_e state, state_n;
  logic [15:0] i_row;
  logic [15:0] s_tok;
  logic [15:0] d_dim;
  logic [63:0] checksum;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) state <= IDLE; else state <= state_n;
  end

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      i_row <= '0; s_tok <= '0; d_dim <= '0; checksum <= 64'd0; checksum_out <= 64'd0;
    end else begin
      unique case (state)
        IDLE: if (start) begin i_row <= '0; s_tok <= '0; d_dim <= '0; checksum <= 64'd0; checksum_out <= 64'd0; end
        RUN: begin
          // dummy math: combine row/token/dim counts into checksum
          checksum <= checksum + i_row + s_tok + d_dim;
          if (d_dim + 1 < head_dim_d) begin
            d_dim <= d_dim + 16'd1;
          end else begin
            d_dim <= 16'd0;
            if (s_tok + 1 < s_tokens) begin
              s_tok <= s_tok + 16'd1;
            end else begin
              s_tok <= 16'd0;
              if (i_row + 1 < m_rows) i_row <= i_row + 16'd1;
            end
          end
        end
        default: ;
      endcase
    end
  end

  always_comb begin
    state_n = state;
    unique case (state)
      IDLE: if (start) state_n = RUN;
      RUN:  if ( (i_row + 1 >= m_rows) && (s_tok + 1 >= s_tokens) && (d_dim + 1 >= head_dim_d)) state_n = DONE;
      DONE: begin checksum_out = checksum; state_n = IDLE; end
      default: state_n = IDLE;
    endcase
  end

endmodule


