// SelectSpec pass: choose a sparse attention spec and set `spec` attribute

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {
namespace {
struct SelectSpecPass : public PassWrapper<SelectSpecPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      // If user specified spec, keep it
      if (op->getAttr("spec")) return;
      // Simple heuristic: prefer sliding_window if window_size present
      if (op->getAttr("window_size")) {
        op->setAttr("spec", StringAttr::get(op->getContext(), "sliding_window"));
        return;
      }
      // Default to BSR (blocked-sparse)
      op->setAttr("spec", StringAttr::get(op->getContext(), "bsr"));
    });
  }
};
} // namespace

std::unique_ptr<Pass> createSelectSpecPass() { return std::make_unique<SelectSpecPass>(); }

} // namespace mlir::sattn



