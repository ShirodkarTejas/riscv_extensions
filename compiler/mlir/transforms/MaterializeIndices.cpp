// Materialize sparse indices (BSR) flag on sattn.sparse_attention ops.

#include "compiler/mlir/transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct MaterializeIndicesPass : public PassWrapper<MaterializeIndicesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "sattn.sparse_attention") {
        // Set a boolean unit attribute to indicate indices are materialized
        op->setAttr("materialized_indices", UnitAttr::get(op->getContext()));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createMaterializeIndicesPass() { return std::make_unique<MaterializeIndicesPass>(); }

} // namespace mlir::sattn


