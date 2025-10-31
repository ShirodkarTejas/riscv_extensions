// Simple dual-port scratchpad: one write, one read

module spad #(
  parameter ADDR_WIDTH = 8,  // depth = 2^ADDR_WIDTH
  parameter DATA_WIDTH = 32
)(
  input  logic                   clk,
  // write port
  input  logic                   wen,
  input  logic [ADDR_WIDTH-1:0]  waddr,
  input  logic [DATA_WIDTH-1:0]  wdata,
  // read port
  input  logic [ADDR_WIDTH-1:0]  raddr,
  output logic [DATA_WIDTH-1:0]  rdata
);

  localparam int DEPTH = (1 << ADDR_WIDTH);
  logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wen) mem[waddr] <= wdata;
  end

  assign rdata = mem[raddr];

endmodule


