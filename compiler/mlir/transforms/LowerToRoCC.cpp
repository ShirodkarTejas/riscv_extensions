// Lower to RoCC: replace sattn.sparse_attention with sattn.rocc_call carrying descriptor attrs

#include "compiler/mlir/transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct LowerToRoCCPass : public PassWrapper<LowerToRoCCPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation*, 8> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      OpBuilder b(op);
      OperationState st(op->getLoc(), "sattn.rocc_call");
      // Copy relevant attributes into the call op
      for (auto &named : op->getAttrs()) {
        st.addAttribute(named.getName(), named.getValue());
      }
      // For convenience, mirror common sizes
      auto tileM = op->getAttr("tile_M");
      auto tileD = op->getAttr("tile_D");
      auto tileS = op->getAttr("tile_S");
      if (tileM) st.addAttribute("m_rows", tileM);
      if (tileD) st.addAttribute("head_dim_d", tileD);
      if (tileS) st.addAttribute("s_tokens", tileS);
      b.create(st);
      toErase.push_back(op);
    });
    for (Operation* op : toErase) op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToRoCCPass() { return std::make_unique<LowerToRoCCPass>(); }

} // namespace mlir::sattn


