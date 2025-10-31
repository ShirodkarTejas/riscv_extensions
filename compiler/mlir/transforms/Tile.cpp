// Tile pass: set simple tiling hints on sattn.sparse_attention ops

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct TilePass : public PassWrapper<TilePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      // Heuristic tiling defaults; real pass would analyze shapes
      op->setAttr("tile_M", IntegerAttr::get(IntegerType::get(op->getContext(), 64), 64));
      op->setAttr("tile_D", IntegerAttr::get(IntegerType::get(op->getContext(), 64), 64));
      op->setAttr("tile_S", IntegerAttr::get(IntegerType::get(op->getContext(), 64), 128));
    });
  }
};
} // namespace

std::unique_ptr<Pass> createTilePass() { return std::make_unique<TilePass>(); }

} // namespace mlir::sattn


