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
      auto *ctx = op->getContext();
      auto toI64 = [&](Attribute a, int64_t def) -> int64_t {
        if (auto ia = dyn_cast_or_null<IntegerAttr>(a)) return ia.getInt();
        return def;
      };
      auto toF64 = [&](Attribute a, double def) -> double {
        if (auto fa = dyn_cast_or_null<FloatAttr>(a)) return fa.getValueAsDouble();
        return def;
      };
      // Read basic tile sizes and optional knobs
      int64_t S = toI64(op->getAttr("tile_S"), 128);
      int64_t D = toI64(op->getAttr("tile_D"), 64);
      int64_t W = toI64(op->getAttr("window_size"), 0);
      int64_t BS = toI64(op->getAttr("block_size"), 64);
      double keep = toF64(op->getAttr("keep_ratio"), 0.12);
      if (BS <= 0) BS = 64;
      // Very simple density model
      double density_bsr = keep; // assume fraction of blocks kept
      double relSpan = (W > 0) ? double(2 * W + 1) / double(std::max<int64_t>(1, S)) : 1.0;
      // Simple threshold: prefer sliding_window when span <= 50% of S
      StringRef spec = (W > 0 && relSpan <= 0.5) ? StringRef("sliding_window") : StringRef("bsr");
      op->setAttr("spec", StringAttr::get(ctx, spec));
    });
  }
};
} // namespace

std::unique_ptr<Pass> createSelectSpecPass() { return std::make_unique<SelectSpecPass>(); }

} // namespace mlir::sattn



