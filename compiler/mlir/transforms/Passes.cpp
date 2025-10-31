#include "compiler/mlir/transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sattn {

void registerPasses() {
  PassRegistration<Pass>("sattn-materialize-indices", "Materialize sparse indices",
                         [] { return createMaterializeIndicesPass(); });
  PassRegistration<Pass>("sattn-tile", "Set tiling hints",
                         [] { return createTilePass(); });
  PassRegistration<Pass>("sattn-fuse-softmax", "Fuse scale+mask+softmax",
                         [] { return createFuseSoftmaxPass(); });
  PassRegistration<Pass>("sattn-lower-to-rvv", "Lower to RVV backend",
                         [] { return createLowerToRVVPass(); });
  PassRegistration<Pass>("sattn-lower-to-rocc", "Lower to RoCC backend",
                         [] { return createLowerToRoCCPass(); });
  PassRegistration<Pass>("sattn-vectorize", "Vectorize RVV call",
                         [] { return createVectorizePass(); });
  PassRegistration<Pass>("sattn-bufferize", "Bufferize calls",
                         [] { return createBufferizePass(); });
}

} // namespace sattn
} // namespace mlir


