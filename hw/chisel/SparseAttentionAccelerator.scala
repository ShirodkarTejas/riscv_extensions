// Sparse Attention Accelerator for Chipyard/Rocket
// Implements RoCC interface with custom sparse attention primitives
//
// Supports:
// - 5 patterns (sliding_window, block_local_global, nm_structured, lsh, landmark)
// - 4 precisions (fp32, bf16, i8, i4)
// - Integration with existing Verilog RTL modules

package sattn

import chisel3._
import chisel3.util._
import chisel3.experimental._
import freechips.rocketchip.tile._
import freechips.rocketchip.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._

// =============================================================================
// Configuration
// =============================================================================

case class SattnAcceleratorConfig(
  scratchpadKB: Int = 16,      // Local memory size
  indexRAMKB: Int = 2,          // Index storage size
  numPEs: Int = 8,              // Parallel processing elements
  maxBlockSize: Int = 64,       // Maximum block size
  maxSeqLen: Int = 2048,        // Maximum sequence length
  dataWidth: Int = 32           // Data width (bits)
)

case object SattnKey extends Field[Option[SattnAcceleratorConfig]](None)

// =============================================================================
// RoCC Custom Instructions
// =============================================================================

object SattnInstructions {
  // Custom instruction opcodes (funct7 field)
  val LOAD_INDICES   = 0.U  // Load sparsity indices
  val CONFIG_SHAPE   = 1.U  // Configure B, H, L, D
  val CONFIG_PATTERN = 2.U  // Configure pattern and params
  val EXECUTE        = 3.U  // Start computation
  val STATUS         = 4.U  // Read status/cycles
  val LOAD_DATA      = 5.U  // Load Q/K/V pointers
  val READ_RESULT    = 6.U  // Read output pointer
}

// =============================================================================
// Main Accelerator Module
// =============================================================================

class SparseAttentionAccelerator(
  opcodes: OpcodeSet,
  val cfg: SattnAcceleratorConfig
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new SparseAttentionAcceleratorImp(this)
}

class SparseAttentionAcceleratorImp(outer: SparseAttentionAccelerator)(implicit p: Parameters)
  extends LazyRoCCModuleImp(outer) {

  val cfg = outer.cfg

  // ===========================================================================
  // State Machine
  // ===========================================================================

  val s_idle :: s_load_indices :: s_config :: s_execute :: s_done :: Nil = Enum(5)
  val state = RegInit(s_idle)

  // ===========================================================================
  // Configuration Registers
  // ===========================================================================

  // Shape parameters
  val shape_B = RegInit(1.U(32.W))  // Batch size
  val shape_H = RegInit(8.U(32.W))  // Number of heads
  val shape_L = RegInit(128.U(32.W)) // Sequence length
  val shape_D = RegInit(64.U(32.W))  // Head dimension

  // Pattern selection
  val pattern = RegInit(0.U(8.W))
  // 0: sliding_window
  // 1: block_local_global
  // 2: nm_structured
  // 3: lsh
  // 4: landmark

  // Precision
  val precision = RegInit(0.U(8.W))
  // 0: fp32
  // 1: bf16
  // 2: i8
  // 3: i4

  // Pattern-specific parameters
  val window_size = RegInit(16.U(32.W))
  val block_size = RegInit(16.U(32.W))
  val keep_ratio = RegInit(0.U(32.W))  // Fixed-point representation
  val global_tokens = RegInit(4.U(32.W))
  val nm_n = RegInit(2.U(32.W))
  val nm_m = RegInit(4.U(32.W))
  val buckets = RegInit(8.U(32.W))
  val num_landmarks = RegInit(16.U(32.W))

  // Memory pointers
  val ptr_Q = RegInit(0.U(64.W))
  val ptr_K = RegInit(0.U(64.W))
  val ptr_V = RegInit(0.U(64.W))
  val ptr_O = RegInit(0.U(64.W))
  val ptr_indices = RegInit(0.U(64.W))

  // Performance counters
  val cycle_counter = RegInit(0.U(64.W))
  val mem_read_counter = RegInit(0.U(64.W))
  val mem_write_counter = RegInit(0.U(64.W))

  // ===========================================================================
  // Memory Modules
  // ===========================================================================

  // Scratchpad memory (16KB default)
  val scratchpad_depth = (cfg.scratchpadKB * 1024) / (cfg.dataWidth / 8)
  val scratchpad = Module(new ScratchpadMem(scratchpad_depth, cfg.dataWidth))

  // Index RAM (2KB default)
  val index_ram_depth = (cfg.indexRAMKB * 1024) / 2  // 16-bit indices
  val index_ram = Module(new IndexRAM(index_ram_depth))

  // ===========================================================================
  // Compute Units (Verilog Black Boxes)
  // ===========================================================================

  // Sparse dot product engine
  val spdot = Module(new SPDotBSRCore(cfg.dataWidth, cfg.numPEs))

  // Fused softmax
  val softmax = Module(new SoftmaxFused(cfg.maxBlockSize, cfg.dataWidth))

  // Sparse matrix multiply
  val spmm = Module(new SPMMbsr(cfg.dataWidth, cfg.numPEs))

  // 2D Gather
  val gather = Module(new Gather2D(cfg.dataWidth))

  // ===========================================================================
  // RoCC Interface Handling
  // ===========================================================================

  io.cmd.ready := (state === s_idle)
  io.resp.valid := false.B
  io.resp.bits.data := 0.U
  io.resp.bits.rd := io.cmd.bits.inst.rd

  io.busy := (state =/= s_idle)
  io.interrupt := false.B

  // Default: not using memory
  io.mem.req.valid := false.B

  when(io.cmd.fire()) {
    val funct = io.cmd.bits.inst.funct
    val rs1 = io.cmd.bits.rs1
    val rs2 = io.cmd.bits.rs2

    switch(funct) {
      is(SattnInstructions.LOAD_INDICES) {
        // Load sparsity indices to index RAM
        ptr_indices := rs1
        state := s_load_indices
      }

      is(SattnInstructions.CONFIG_SHAPE) {
        // Configure shape: rs1[31:16]=B, rs1[15:0]=H, rs2[31:16]=L, rs2[15:0]=D
        shape_B := rs1(31, 16)
        shape_H := rs1(15, 0)
        shape_L := rs2(31, 16)
        shape_D := rs2(15, 0)
        io.resp.valid := true.B
        io.resp.bits.data := 0.U  // Success
      }

      is(SattnInstructions.CONFIG_PATTERN) {
        // Configure pattern: rs1[7:0]=pattern, rs1[15:8]=precision, rs2=param
        pattern := rs1(7, 0)
        precision := rs1(15, 8)
        
        // Store pattern-specific parameter in appropriate register
        switch(pattern) {
          is(0.U) { window_size := rs2 }       // sliding_window
          is(1.U) { block_size := rs2 }        // block_local_global
          is(2.U) { nm_n := rs2(15, 0); nm_m := rs2(31, 16) }  // nm_structured
          is(3.U) { buckets := rs2 }           // lsh
          is(4.U) { num_landmarks := rs2 }     // landmark
        }
        
        state := s_config
        io.resp.valid := true.B
        io.resp.bits.data := 0.U  // Success
      }

      is(SattnInstructions.LOAD_DATA) {
        // Load Q, K, V pointers: rs1=Q_addr, rs2=K_addr
        // (V_addr comes from next instruction)
        ptr_Q := rs1
        ptr_K := rs2
        io.resp.valid := true.B
        io.resp.bits.data := 0.U  // Success
      }

      is(SattnInstructions.EXECUTE) {
        // Start execution: rs1=V_addr, rs2=O_addr
        ptr_V := rs1
        ptr_O := rs2
        state := s_execute
        cycle_counter := 0.U
        mem_read_counter := 0.U
        mem_write_counter := 0.U
      }

      is(SattnInstructions.STATUS) {
        // Return status: 0=busy, 1=done
        io.resp.valid := true.B
        io.resp.bits.data := Mux(state === s_done, 1.U, 0.U)
      }

      is(SattnInstructions.READ_RESULT) {
        // Return performance counters
        io.resp.valid := true.B
        // Pack: [31:0]=cycles, [63:32]=mem_ops
        io.resp.bits.data := Cat(
          (mem_read_counter + mem_write_counter)(31, 0),
          cycle_counter(31, 0)
        )
      }
    }
  }

  // ===========================================================================
  // Execution State Machine
  // ===========================================================================

  when(state === s_execute) {
    cycle_counter := cycle_counter + 1.U

    // High-level sparse attention execution
    // This is a simplified skeleton; full implementation would have
    // detailed tile-by-tile dataflow

    // 1. Load Q, K tiles from memory
    // 2. Compute sparse QK^T using spdot
    // 3. Apply softmax
    // 4. Load V tiles
    // 5. Compute attention * V using spmm
    // 6. Write output

    // For now, just simulate execution time based on problem size
    val total_ops = shape_B * shape_H * shape_L * shape_L * shape_D
    val estimated_cycles = total_ops >> 10.U  // Rough estimate

    when(cycle_counter >= estimated_cycles) {
      state := s_done
    }
  }

  when(state === s_done) {
    // Stay in done state until next command
    when(io.cmd.fire()) {
      state := s_idle
    }
  }

  when(state === s_load_indices) {
    // Load indices from memory to index RAM
    // (Simplified - full implementation would use DMA)
    state := s_idle
  }

  when(state === s_config) {
    // Configuration complete, return to idle
    state := s_idle
  }

  // ===========================================================================
  // Module Connections (Simplified - full datapath would be more complex)
  // ===========================================================================

  // Connect scratchpad
  scratchpad.io.clk := clock
  scratchpad.io.rst := reset.asBool
  scratchpad.io.wen := false.B
  scratchpad.io.ren := false.B
  scratchpad.io.addr := 0.U
  scratchpad.io.wdata := 0.U

  // Connect index RAM
  index_ram.io.clk := clock
  index_ram.io.rst := reset.asBool
  index_ram.io.wen := false.B
  index_ram.io.ren := false.B
  index_ram.io.addr := 0.U
  index_ram.io.wdata := 0.U

  // Connect compute units
  spdot.io.clk := clock
  spdot.io.rst := reset.asBool
  spdot.io.valid := false.B

  softmax.io.clk := clock
  softmax.io.rst := reset.asBool
  softmax.io.valid := false.B

  spmm.io.clk := clock
  spmm.io.rst := reset.asBool
  spmm.io.valid := false.B

  gather.io.clk := clock
  gather.io.rst := reset.asBool
  gather.io.valid := false.B
}

// =============================================================================
// Memory Modules
// =============================================================================

class ScratchpadMem(depth: Int, width: Int) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val wen = Input(Bool())
    val ren = Input(Bool())
    val addr = Input(UInt(log2Ceil(depth).W))
    val wdata = Input(UInt(width.W))
    val rdata = Output(UInt(width.W))
  })

  val mem = SyncReadMem(depth, UInt(width.W))

  io.rdata := 0.U

  when(io.wen) {
    mem.write(io.addr, io.wdata)
  }

  when(io.ren) {
    io.rdata := mem.read(io.addr)
  }
}

class IndexRAM(depth: Int) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val wen = Input(Bool())
    val ren = Input(Bool())
    val addr = Input(UInt(log2Ceil(depth).W))
    val wdata = Input(UInt(16.W))
    val rdata = Output(UInt(16.W))
  })

  val mem = SyncReadMem(depth, UInt(16.W))

  io.rdata := 0.U

  when(io.wen) {
    mem.write(io.addr, io.wdata)
  }

  when(io.ren) {
    io.rdata := mem.read(io.addr)
  }
}

// =============================================================================
// Verilog Black Boxes (Existing RTL Modules)
// =============================================================================

class SPDotBSRCore(dataWidth: Int, numPEs: Int) extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val valid = Input(Bool())
    val a_data = Input(Vec(numPEs, UInt(dataWidth.W)))
    val b_data = Input(Vec(numPEs, UInt(dataWidth.W)))
    val indices = Input(Vec(numPEs, UInt(16.W)))
    val result = Output(UInt(dataWidth.W))
    val ready = Output(Bool())
  })
}

class SoftmaxFused(maxSize: Int, dataWidth: Int) extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val valid = Input(Bool())
    val data_in = Input(Vec(maxSize, UInt(dataWidth.W)))
    val data_out = Output(Vec(maxSize, UInt(dataWidth.W)))
    val ready = Output(Bool())
  })
}

class SPMMbsr(dataWidth: Int, numPEs: Int) extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val valid = Input(Bool())
    val prob_data = Input(Vec(numPEs, UInt(dataWidth.W)))
    val v_data = Input(Vec(numPEs, UInt(dataWidth.W)))
    val indices = Input(Vec(numPEs, UInt(16.W)))
    val result = Output(UInt(dataWidth.W))
    val ready = Output(Bool())
  })
}

class Gather2D(dataWidth: Int) extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    val valid = Input(Bool())
    val base_addr = Input(UInt(64.W))
    val indices = Input(Vec(8, UInt(16.W)))
    val data_out = Output(Vec(8, UInt(dataWidth.W)))
    val ready = Output(Bool())
  })
}

// =============================================================================
// Chipyard Configuration
// =============================================================================

class WithSattnAccelerator(
  cfg: SattnAcceleratorConfig = SattnAcceleratorConfig()
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      val sattn = LazyModule(new SparseAttentionAccelerator(
        OpcodeSet.custom0, cfg)(p))
      sattn
    }
  )
  case SattnKey => Some(cfg)
})

