// Lower to RVV: replace sattn.sparse_attention with sattn.rvv_call carrying tile attrs

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct LowerToRVVPass : public PassWrapper<LowerToRVVPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation*, 8> toErase;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      OpBuilder b(op);
      OperationState st(op->getLoc(), "sattn.rvv_call");
      // Copy relevant attributes
      for (auto &named : op->getAttrs()) st.addAttribute(named.getName(), named.getValue());
      // Mirror tiling to common names
      if (auto a = op->getAttr("tile_M")) st.addAttribute("m_rows", a);
      if (auto a = op->getAttr("tile_D")) st.addAttribute("head_dim_d", a);
      if (auto a = op->getAttr("tile_S")) st.addAttribute("s_tokens", a);
      // Per-spec hook: mark BLG lowering toggle so backends can specialize
      if (auto spec = dyn_cast_or_null<StringAttr>(op->getAttr("spec"))) {
        if (spec.getValue() == "block_local_global") {
          st.addAttribute("blg_enabled", BoolAttr::get(op->getContext(), true));
        }
        if (spec.getValue() == "nm_structured") {
          st.addAttribute("nm_enabled", BoolAttr::get(op->getContext(), true));
        }
        if (spec.getValue() == "topk_per_query") {
          st.addAttribute("topk_enabled", BoolAttr::get(op->getContext(), true));
        }
        if (spec.getValue() == "lsh") {
          st.addAttribute("lsh_enabled", BoolAttr::get(op->getContext(), true));
        }
      }
      b.create(st);
      toErase.push_back(op);
    });
    for (Operation* op : toErase) op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToRVVPass() { return std::make_unique<LowerToRVVPass>(); }

} // namespace mlir::sattn


