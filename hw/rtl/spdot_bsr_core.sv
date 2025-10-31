// Simple spdot_bsr core skeleton
// Iterates over m_rows x s_tokens x head_dim cycles to emulate MAC work
// Does not access external memory; intended as a compute loop placeholder

module spdot_bsr_core (
  input  logic        clk,
  input  logic        rstn,
  input  logic        start,
  input  logic [15:0] m_rows,
  input  logic [15:0] head_dim_d,
  input  logic [15:0] s_tokens,
  // scratchpad read ports
  output logic [15:0] q_raddr,
  input  logic [31:0] q_rdata,
  output logic [15:0] k_raddr,
  input  logic [31:0] k_rdata,
  output logic        busy,
  output logic        done,
  output logic [63:0] checksum_out
);

  typedef enum logic [1:0] {IDLE, RUN, DONE} state_e;
  state_e state, state_n;

  logic [31:0] i_row;
  logic [31:0] j_tok;
  logic [31:0] k_dim;

  // Accumulator for dot product over head_dim
  logic [63:0] acc;
  logic [63:0] checksum;

  assign busy = (state == RUN);
  assign done = (state == DONE);

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) state <= IDLE; else state <= state_n;
  end

  // Initialize local buffers on start and run a MAC per cycle over k_dim
  integer idx;
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      i_row <= 32'd0; j_tok <= 32'd0; k_dim <= 32'd0; acc <= 64'd0; checksum <= 64'd0; checksum_out <= 64'd0;
      q_raddr <= 16'd0; k_raddr <= 16'd0;
    end else begin
      unique case (state)
        IDLE: begin
          if (start) begin
            i_row <= 32'd0; j_tok <= 32'd0; k_dim <= 32'd0; acc <= 64'd0; checksum <= 64'd0; checksum_out <= 64'd0;
            q_raddr <= 16'd0; k_raddr <= 16'd0;
          end
        end
        RUN: begin
          // Perform one MAC per cycle: acc += q[k]*k[k]
          if (k_dim < head_dim_d) begin
            acc <= acc + (q_rdata * k_rdata);
          end
          // advance innermost dimension first
          if (k_dim + 1 < head_dim_d) begin
            k_dim <= k_dim + 32'd1;
            q_raddr <= k_dim + 16'd1;
            k_raddr <= k_dim + 16'd1;
          end else begin
            // dot for this token finished; fold into checksum, reset acc and move to next token
            checksum <= checksum + acc;
            acc <= 64'd0;
            k_dim <= 32'd0;
            q_raddr <= 16'd0;
            k_raddr <= 16'd0;
            if (j_tok + 1 < s_tokens) begin
              j_tok <= j_tok + 32'd1;
            end else begin
              j_tok <= 32'd0;
              if (i_row + 1 < m_rows) begin
                i_row <= i_row + 32'd1;
              end
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
      RUN:  if ( (i_row + 1 >= m_rows) && (j_tok + 1 >= s_tokens) && (k_dim + 1 >= head_dim_d)) begin
               state_n = DONE;
             end
      DONE: begin
               // latch final checksum to output
               checksum_out = checksum;
               state_n = IDLE;
             end
      default: state_n = IDLE;
    endcase
  end

endmodule


