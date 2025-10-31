#include "transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sattn {

void registerPasses() {
  registerPass("sattn-materialize-indices", "Materialize sparse indices",
               [] { return createMaterializeIndicesPass(); });
  registerPass("sattn-tile", "Set tiling hints",
               [] { return createTilePass(); });
  registerPass("sattn-fuse-softmax", "Fuse scale+mask+softmax",
               [] { return createFuseSoftmaxPass(); });
  registerPass("sattn-lower-to-rvv", "Lower to RVV backend",
               [] { return createLowerToRVVPass(); });
  registerPass("sattn-lower-to-rocc", "Lower to RoCC backend",
               [] { return createLowerToRoCCPass(); });
  registerPass("sattn-vectorize", "Vectorize RVV call",
               [] { return createVectorizePass(); });
  registerPass("sattn-bufferize", "Bufferize calls",
               [] { return createBufferizePass(); });
}

} // namespace sattn
} // namespace mlir


