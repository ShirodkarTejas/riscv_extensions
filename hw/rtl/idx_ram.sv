// Simple index RAM: one write port (MMIO), one read port (gather)

module idx_ram #(
  parameter ADDR_WIDTH = 10, // depth = 2^ADDR_WIDTH
  parameter DATA_WIDTH = 16
)(
  input  logic                   clk,
  input  logic                   rstn,
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

  // simple read (registered or comb). Use comb for simplicity.
  assign rdata = mem[raddr];

endmodule


