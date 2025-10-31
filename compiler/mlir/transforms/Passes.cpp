#include "transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sattn {

void registerPasses() {}

// Expose pipelines instead of individual pass flags for stability across MLIR
static mlir::PassPipelineRegistration<> SattnLowerRoCC(
    "sattn-lower-rocc",
    "Lower sattn.sparse_attention to sattn.rocc_call with materialized indices, tiling, and softmax fusion",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createSelectSpecPass());
      pm.addPass(createMaterializeIndicesPass());
      pm.addPass(createTilePass());
      pm.addPass(createFuseSoftmaxPass());
      pm.addPass(createLowerToRoCCPass());
      pm.addPass(createBufferizePass());
    });

static mlir::PassPipelineRegistration<> SattnLowerRVV(
    "sattn-lower-rvv",
    "Annotate RVV path with vectorize+bufferize (placeholder)",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createVectorizePass());
      pm.addPass(createBufferizePass());
    });

} // namespace sattn
} // namespace mlir


