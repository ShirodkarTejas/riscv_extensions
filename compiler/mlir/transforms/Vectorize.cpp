// Vectorize pass: annotate rvv_call with vectorized=true as a placeholder

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {
namespace {
struct VectorizePass : public PassWrapper<VectorizePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "sattn.rvv_call") {
        op->setAttr("vectorized", UnitAttr::get(op->getContext()));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createVectorizePass() { return std::make_unique<VectorizePass>(); }

} // namespace mlir::sattn


