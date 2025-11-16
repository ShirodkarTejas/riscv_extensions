// RISC-V Custom Instruction Decoder for Sparse Attention Accelerator
// Decodes custom-1 opcode (0x2B) instructions and extracts command fields

module sattn_inst_decode #(
  parameter INST_WIDTH = 32
)(
  input  logic                    clk,
  input  logic                    rstn,
  
  // Instruction interface
  input  logic                    inst_valid,     // Instruction valid
  input  logic [INST_WIDTH-1:0]   inst_bits,      // 32-bit instruction word
  output logic                    inst_ready,     // Ready to accept instruction
  
  // Decoded outputs
  output logic                    decode_valid,   // Decoded instruction valid
  output logic [2:0]              primitive,      // Which primitive (funct3)
  output logic [6:0]              funct7,         // Function variant
  output logic [4:0]              rs1,            // Source register 1
  output logic [4:0]              rs2,            // Source register 2
  output logic [4:0]              rd,             // Destination register
  output logic                    illegal_inst    // Illegal instruction flag
);

  // Instruction format fields (R-type)
  localparam OPCODE_LSB  = 0;
  localparam OPCODE_MSB  = 6;
  localparam RD_LSB      = 7;
  localparam RD_MSB      = 11;
  localparam FUNCT3_LSB  = 12;
  localparam FUNCT3_MSB  = 14;
  localparam RS1_LSB     = 15;
  localparam RS1_MSB     = 19;
  localparam RS2_LSB     = 20;
  localparam RS2_MSB     = 24;
  localparam FUNCT7_LSB  = 25;
  localparam FUNCT7_MSB  = 31;
  
  // Expected opcode for custom-1
  localparam SATTN_OPCODE = 7'b0101011;  // 0x2B
  
  // Valid primitives (funct3 values)
  localparam FUNCT3_BLK_REDUCE    = 3'b000;
  localparam FUNCT3_TOPK_IDX      = 3'b001;
  localparam FUNCT3_GATH2D        = 3'b010;
  localparam FUNCT3_SCAT2D        = 3'b011;
  localparam FUNCT3_SPDOT_BSR     = 3'b100;
  localparam FUNCT3_SOFTMAX_FUSED = 3'b101;
  localparam FUNCT3_SPMM_BSR      = 3'b110;
  localparam FUNCT3_RESERVED      = 3'b111;
  
  // Valid funct7 values (currently only 0x00)
  localparam FUNCT7_DEFAULT = 7'b0000000;
  
  // Extract instruction fields
  wire [6:0] opcode_field  = inst_bits[OPCODE_MSB:OPCODE_LSB];
  wire [4:0] rd_field      = inst_bits[RD_MSB:RD_LSB];
  wire [2:0] funct3_field  = inst_bits[FUNCT3_MSB:FUNCT3_LSB];
  wire [4:0] rs1_field     = inst_bits[RS1_MSB:RS1_LSB];
  wire [4:0] rs2_field     = inst_bits[RS2_MSB:RS2_LSB];
  wire [6:0] funct7_field  = inst_bits[FUNCT7_MSB:FUNCT7_LSB];
  
  // Opcode match
  wire opcode_match = (opcode_field == SATTN_OPCODE);
  
  // funct7 validation (only 0x00 currently valid)
  wire funct7_valid = (funct7_field == FUNCT7_DEFAULT);
  
  // funct3 validation (0-6 valid, 7 reserved)
  wire funct3_valid = (funct3_field != FUNCT3_RESERVED);
  
  // Instruction is legal if opcode matches, funct7 valid, and funct3 valid
  wire inst_legal = opcode_match && funct7_valid && funct3_valid;
  
  // Decode state machine
  typedef enum logic [1:0] {
    IDLE,
    DECODE,
    DONE
  } state_e;
  
  state_e state, state_n;
  
  // Decoded output registers
  logic [2:0]  primitive_r;
  logic [6:0]  funct7_r;
  logic [4:0]  rs1_r, rs2_r, rd_r;
  logic        illegal_inst_r;
  logic        decode_valid_r;
  
  // FSM: simple 1-cycle decode
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      state <= IDLE;
      primitive_r <= 3'b0;
      funct7_r <= 7'b0;
      rs1_r <= 5'b0;
      rs2_r <= 5'b0;
      rd_r <= 5'b0;
      illegal_inst_r <= 1'b0;
      decode_valid_r <= 1'b0;
    end else begin
      state <= state_n;
      
      case (state)
        IDLE: begin
          decode_valid_r <= 1'b0;
          if (inst_valid) begin
            // Capture and decode in one cycle
            primitive_r <= funct3_field;
            funct7_r <= funct7_field;
            rs1_r <= rs1_field;
            rs2_r <= rs2_field;
            rd_r <= rd_field;
            illegal_inst_r <= !inst_legal;
            decode_valid_r <= 1'b1;
          end
        end
        
        DECODE: begin
          // Hold decoded values
          decode_valid_r <= 1'b1;
        end
        
        DONE: begin
          decode_valid_r <= 1'b0;
        end
        
        default: begin
          decode_valid_r <= 1'b0;
        end
      endcase
    end
  end
  
  // Next state logic
  always_comb begin
    state_n = state;
    
    case (state)
      IDLE: begin
        if (inst_valid) begin
          state_n = DECODE;
        end
      end
      
      DECODE: begin
        // Move to done after 1 cycle
        state_n = DONE;
      end
      
      DONE: begin
        state_n = IDLE;
      end
      
      default: state_n = IDLE;
    endcase
  end
  
  // Outputs
  assign decode_valid = decode_valid_r;
  assign primitive    = primitive_r;
  assign funct7       = funct7_r;
  assign rs1          = rs1_r;
  assign rs2          = rs2_r;
  assign rd           = rd_r;
  assign illegal_inst = illegal_inst_r;
  assign inst_ready   = (state == IDLE);
  
  // Assertions for verification (optional, synthesis tools may ignore)
  /* verilator lint_off UNUSED */
  logic unused_ok;
  assign unused_ok = &{1'b0, inst_valid, inst_bits};
  /* verilator lint_on UNUSED */
  
  `ifdef FORMAL
    // Formal verification properties
    
    // Property: Valid decode implies opcode matched
    assert property (@(posedge clk) disable iff (!rstn)
      decode_valid && !illegal_inst |-> opcode_field == SATTN_OPCODE);
    
    // Property: Illegal instruction set when opcode mismatches
    assert property (@(posedge clk) disable iff (!rstn)
      decode_valid && (opcode_field != SATTN_OPCODE) |-> illegal_inst);
    
    // Property: Reserved funct3 (0x7) causes illegal instruction
    assert property (@(posedge clk) disable iff (!rstn)
      decode_valid && (funct3_field == FUNCT3_RESERVED) |-> illegal_inst);
      
    // Property: Ready goes low when instruction accepted
    assert property (@(posedge clk) disable iff (!rstn)
      inst_valid && inst_ready |=> !inst_ready);
  `endif

endmodule

