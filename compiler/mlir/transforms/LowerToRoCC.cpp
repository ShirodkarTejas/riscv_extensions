// Lower to RoCC: annotate sattn.sparse_attention ops with lowered_backend="rocc"

#include "compiler/mlir/transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct LowerToRoCCPass : public PassWrapper<LowerToRoCCPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "sattn.sparse_attention") {
        op->setAttr("lowered_backend", StringAttr::get(op->getContext(), "rocc"));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToRoCCPass() { return std::make_unique<LowerToRoCCPass>(); }

} // namespace mlir::sattn


