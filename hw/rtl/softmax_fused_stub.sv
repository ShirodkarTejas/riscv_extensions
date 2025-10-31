// softmax_fused stub: consumes M rows by S tokens and produces a checksum-like output
// This is a placeholder for a fused softmax pipeline.

module softmax_fused_stub (
  input  logic        clk,
  input  logic        rstn,
  input  logic        start,
  input  logic [15:0] m_rows,
  input  logic [15:0] s_tokens,
  output logic        busy,
  output logic        done,
  output logic [63:0] checksum_out
);

  typedef enum logic [1:0] {IDLE, RUN, DONE} state_e;
  state_e state, state_n;
  logic [15:0] i_row;
  logic [15:0] s_tok;
  logic [63:0] checksum;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) state <= IDLE; else state <= state_n;
  end

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      i_row <= '0; s_tok <= '0; checksum <= 64'd0; checksum_out <= 64'd0;
    end else begin
      // verilator lint_off CASEINCOMPLETE
      case (state)
        IDLE: if (start) begin i_row <= '0; s_tok <= '0; checksum <= 64'd0; checksum_out <= 64'd0; end
        RUN: begin
          // dummy math: add row+token to checksum (widen to 64b)
          checksum <= checksum + {{48{1'b0}}, i_row} + {{48{1'b0}}, s_tok};
          if (s_tok + 1 < s_tokens) begin
            s_tok <= s_tok + 16'd1;
          end else begin
            s_tok <= 16'd0;
            if (i_row + 1 < m_rows) i_row <= i_row + 16'd1;
          end
        end
        DONE: begin
          // latch final checksum on DONE
          checksum_out <= checksum;
        end
      endcase
      // verilator lint_on CASEINCOMPLETE
    end
  end

  always_comb begin
    state_n = state;
    case (state)
      IDLE: if (start) state_n = RUN;
      RUN:  if ( (i_row + 1 >= m_rows) && (s_tok + 1 >= s_tokens)) state_n = DONE;
      DONE: begin state_n = IDLE; end
      default: state_n = IDLE;
    endcase
  end

endmodule


