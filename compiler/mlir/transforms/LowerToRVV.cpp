// Lower to RVV: annotate sattn.sparse_attention ops with lowered_backend="rvv"

#include "compiler/mlir/transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct LowerToRVVPass : public PassWrapper<LowerToRVVPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "sattn.sparse_attention") {
        op->setAttr("lowered_backend", StringAttr::get(op->getContext(), "rvv"));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToRVVPass() { return std::make_unique<LowerToRVVPass>(); }

} // namespace mlir::sattn


