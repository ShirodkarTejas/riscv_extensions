// Chipyard Configurations for Sparse Attention Accelerator

package sattn

import org.chipsalliance.cde.config._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._

// =============================================================================
// Base Configuration: Rocket + Sparse Attention Accelerator
// =============================================================================

class SattnRocketConfig extends Config(
  new WithSattnAccelerator() ++                    // Add our accelerator
  new WithNBigCores(1) ++                          // 1 Rocket core
  new WithNBanks(1) ++                             // 1 memory bank
  new WithNMemoryChannels(1) ++                    // 1 memory channel
  new WithCacheBlockBytes(64) ++                   // 64B cache blocks
  new freechips.rocketchip.system.BaseConfig       // Base Rocket config
)

// =============================================================================
// Optimized Configuration: Larger caches for sparse attention
// =============================================================================

class SattnRocketOptimizedConfig extends Config(
  new WithSattnAccelerator(
    SattnAcceleratorConfig(
      scratchpadKB = 32,      // Larger scratchpad
      indexRAMKB = 4,         // More index storage
      numPEs = 16,            // More parallelism
      maxBlockSize = 128,     // Larger blocks
      maxSeqLen = 4096        // Longer sequences
    )
  ) ++
  new WithNBigCores(1) ++
  new WithL1ICacheSets(128) ++                     // Larger I$
  new WithL1DCacheSets(128) ++                     // Larger D$
  new WithL1DCacheWays(8) ++                       // More D$ ways
  new freechips.rocketchip.system.BaseConfig
)

// =============================================================================
// BOOM Configuration: Out-of-order + Sparse Attention
// =============================================================================

class SattnBoomConfig extends Config(
  new WithSattnAccelerator() ++
  new boom.common.WithNLargeBooms(1) ++            // 1 BOOM core
  new freechips.rocketchip.system.BaseConfig
)

// =============================================================================
// Multi-Core Configuration: 4 Rockets + Shared Accelerator
// =============================================================================

class SattnMultiCoreConfig extends Config(
  new WithSattnAccelerator() ++
  new WithNBigCores(4) ++                          // 4 Rocket cores
  new WithNBanks(4) ++                             // 4 memory banks
  new WithNMemoryChannels(2) ++                    // 2 memory channels
  new freechips.rocketchip.system.BaseConfig
)

// =============================================================================
// Debug Configuration: Small, fast simulation
// =============================================================================

class SattnDebugConfig extends Config(
  new WithSattnAccelerator(
    SattnAcceleratorConfig(
      scratchpadKB = 4,       // Small for fast simulation
      indexRAMKB = 1,
      numPEs = 4,
      maxBlockSize = 32,
      maxSeqLen = 256
    )
  ) ++
  new WithNBigCores(1) ++
  new WithL1ICacheSets(64) ++
  new WithL1DCacheSets(64) ++
  new freechips.rocketchip.system.BaseConfig
)

