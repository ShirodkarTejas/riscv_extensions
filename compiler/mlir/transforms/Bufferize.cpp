// Bufferize pass: annotate rvv_call/rocc_call with bufferized=true as a placeholder

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {
namespace {
struct BufferizePass : public PassWrapper<BufferizePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "sattn.rvv_call" || name == "sattn.rocc_call") {
        op->setAttr("bufferized", UnitAttr::get(op->getContext()));
        // Basic placeholder for bufferization strategy & layout
        op->setAttr("buffer_strategy", StringAttr::get(op->getContext(), "one-shot"));
        op->setAttr("buffer_layout", StringAttr::get(op->getContext(), "rowmajor"));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createBufferizePass() { return std::make_unique<BufferizePass>(); }

} // namespace mlir::sattn


