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
  input  logic [15:0] block_size,
  // scratchpad read ports (unused in this stub path)
  output logic [15:0] q_raddr,
  input  logic [31:0] q_rdata,
  output logic [15:0] k_raddr,
  input  logic [31:0] k_rdata,
  // index RAM read (provided by top-level via mux)
  output logic [15:0] idx_rd_addr,
  input  logic [15:0] idx_rd_data,
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
  // temporaries for synthetic Q/K generation
  /* verilator lint_off UNUSED */
  logic [31:0] addr32_next;
  logic [31:0] addr32_tok;
  /* verilator lint_on UNUSED */
  /* verilator lint_off UNUSED */
  logic [15:0] block_size_unused;
  assign block_size_unused = block_size;
  /* verilator lint_on UNUSED */

  /* verilator lint_off UNUSED */
  wire [31:0] _unused_qr = q_rdata ^ k_rdata;
  /* verilator lint_on UNUSED */
  assign busy = (state == RUN);
  assign done = (state == DONE);

  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) state <= IDLE; else state <= state_n;
  end

  // Initialize local buffers on start and run a MAC per cycle over k_dim
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
          // advance innermost dimension; include current product on last dim into checksum
          // fetch block id for this token from index RAM and synthesize Q/K per TB
          // compute products inline from idx_rd_data and k_dim
          if (k_dim + 1 < head_dim_d) begin
            acc <= acc + ( {32'd0, {idx_rd_data, k_dim[15:0]}} * {32'd0, {(idx_rd_data ^ 16'h0f0f), k_dim[15:0]}} );
            `ifdef VERILATOR
            if (i_row == 32'd0 && j_tok < 32'd2 && k_dim < 32'd3) begin
              $display("DBG RUN preinc: t=%0d k=%0d idx_addr=%0d idx=%0d q=0x%08x kv=0x%08x acc=0x%016x",
                       j_tok[15:0], k_dim[15:0], idx_rd_addr, idx_rd_data,
                       {idx_rd_data, k_dim[15:0]}, {(idx_rd_data ^ 16'h0f0f), k_dim[15:0]}, acc);
            end
            `endif
            k_dim <= k_dim + 32'd1;
            addr32_next <= (j_tok * head_dim_d) + (k_dim + 32'd1);
            q_raddr <= addr32_next[15:0];
            k_raddr <= addr32_next[15:0];
          end else begin
            // last dim: fold acc plus current product
            checksum <= checksum + (acc + ( {32'd0, {idx_rd_data, k_dim[15:0]}} * {32'd0, {(idx_rd_data ^ 16'h0f0f), k_dim[15:0]}} ));
            `ifdef VERILATOR
            if (i_row == 32'd0 && j_tok < 32'd2) begin
              $display("DBG RUN lastdim: t=%0d k=%0d idx=%0d acc_final=0x%016x",
                       j_tok[15:0], k_dim[15:0], idx_rd_data, checksum + ( {32'd0, {idx_rd_data, k_dim[15:0]}} * {32'd0, {(idx_rd_data ^ 16'h0f0f), k_dim[15:0]}} ));
            end
            `endif
            acc <= 64'd0;
            k_dim <= 32'd0;
            addr32_tok <= ((j_tok + 32'd1) * head_dim_d);
            q_raddr <= (j_tok + 32'd1 < s_tokens) ? addr32_tok[15:0] : 16'd0;
            k_raddr <= (j_tok + 32'd1 < s_tokens) ? addr32_tok[15:0] : 16'd0;
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
        DONE: begin
          // latch final checksum on DONE
          checksum_out <= checksum;
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
               state_n = IDLE;
             end
      default: state_n = IDLE;
    endcase
  end

  // Drive index read address combinationally from current token
  always_comb begin
    idx_rd_addr = (block_size != 16'd0) ? (j_tok[15:0] / block_size) : 16'd0;
  end

endmodule


