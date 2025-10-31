// Fuse softmax: tag sattn.sparse_attention ops with fused_softmax=true when softmax_mode=logsumexp

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct FuseSoftmaxPass : public PassWrapper<FuseSoftmaxPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      if (auto attr = op->getAttrOfType<StringAttr>("softmax_mode")) {
        if (attr.getValue() == "logsumexp") {
          op->setAttr("fused_softmax", UnitAttr::get(op->getContext()));
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createFuseSoftmaxPass() { return std::make_unique<FuseSoftmaxPass>(); }

} // namespace mlir::sattn


