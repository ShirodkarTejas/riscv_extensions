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
    "Lower sattn.sparse_attention to sattn.rvv_call then vectorize+bufferize",
    [](mlir::OpPassManager &pm) {
      pm.addPass(createSelectSpecPass());
      pm.addPass(createLowerToRVVPass());
      pm.addPass(createVectorizePass());
      pm.addPass(createBufferizePass());
    });

// Single-pass pipelines for convenient invocation
static mlir::PassPipelineRegistration<> SattnSelectSpecOnly(
    "sattn-select-spec", "SelectSpec only",
    [](mlir::OpPassManager &pm) { pm.addPass(createSelectSpecPass()); });

static mlir::PassPipelineRegistration<> SattnMaterializeOnly(
    "sattn-materialize-indices", "MaterializeIndices only",
    [](mlir::OpPassManager &pm) { pm.addPass(createMaterializeIndicesPass()); });

static mlir::PassPipelineRegistration<> SattnTileOnly(
    "sattn-tile", "Tile only",
    [](mlir::OpPassManager &pm) { pm.addPass(createTilePass()); });

static mlir::PassPipelineRegistration<> SattnFuseSoftmaxOnly(
    "sattn-fuse-softmax", "FuseSoftmax only",
    [](mlir::OpPassManager &pm) { pm.addPass(createFuseSoftmaxPass()); });

static mlir::PassPipelineRegistration<> SattnLowerToRVVOnly(
    "sattn-lower-to-rvv", "LowerToRVV only",
    [](mlir::OpPassManager &pm) { pm.addPass(createLowerToRVVPass()); });

static mlir::PassPipelineRegistration<> SattnLowerToRoCCOnly(
    "sattn-lower-to-rocc", "LowerToRoCC only",
    [](mlir::OpPassManager &pm) { pm.addPass(createLowerToRoCCPass()); });

static mlir::PassPipelineRegistration<> SattnVectorizeOnly(
    "sattn-vectorize", "Vectorize only",
    [](mlir::OpPassManager &pm) { pm.addPass(createVectorizePass()); });

static mlir::PassPipelineRegistration<> SattnBufferizeOnly(
    "sattn-bufferize", "Bufferize only",
    [](mlir::OpPassManager &pm) { pm.addPass(createBufferizePass()); });

} // namespace sattn
} // namespace mlir


