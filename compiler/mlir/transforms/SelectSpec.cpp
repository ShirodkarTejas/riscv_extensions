// SelectSpec pass: choose a sparse attention spec and set `spec` attribute

#include "transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Process.h"
#include <limits>
#include <algorithm>

using namespace mlir;

namespace mlir::sattn {
namespace {
struct SelectSpecPass : public PassWrapper<SelectSpecPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "sattn.sparse_attention") return;
      // If user specified spec or force_spec, keep it
      if (op->getAttr("spec")) return;
      if (auto force = dyn_cast_or_null<StringAttr>(op->getAttr("force_spec"))) {
        op->setAttr("spec", force);
        return;
      }
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
      // Env override: SATTN_FORCE_SPEC takes precedence
      if (auto envForce = llvm::sys::Process::GetEnv("SATTN_FORCE_SPEC")) {
        op->setAttr("spec", StringAttr::get(ctx, *envForce));
        return;
      }
      // If global tokens requested, prefer block_local_global
      if (op->getAttr("global_tokens")) {
        op->setAttr("spec", StringAttr::get(ctx, "block_local_global"));
        return;
      }
      // If N:M structured attributes are present, prefer nm_structured unless disabled
      if (op->getAttr("nm_n") && op->getAttr("nm_m") &&
          !llvm::sys::Process::GetEnv("SATTN_DISABLE_NM")) {
        op->setAttr("spec", StringAttr::get(ctx, "nm_structured"));
        return;
      }
      // If top-k attribute present, prefer topk_per_query unless disabled
      if (op->getAttr("topk_k") && !llvm::sys::Process::GetEnv("SATTN_DISABLE_TOPK")) {
        op->setAttr("spec", StringAttr::get(ctx, "topk_per_query"));
        return;
      }
      // If LSH buckets present, prefer lsh unless disabled
      if (op->getAttr("lsh_buckets") && !llvm::sys::Process::GetEnv("SATTN_DISABLE_LSH")) {
        op->setAttr("spec", StringAttr::get(ctx, "lsh"));
        return;
      }
      // Density models
      double density_bsr = std::clamp(keep, 0.0, 1.0);
      double relSpan = (W > 0) ? double(2 * W + 1) / double(std::max<int64_t>(1, S)) : 1.0;
      double density_sw = (W > 0) ? std::min<double>(1.0, relSpan) : 1.0;
      // Cache-fit heuristic for BSR tiles: bytes per block of Q/K tiles
      constexpr double L1Bytes = 32.0 * 1024.0;
      double bytesPerBlock = double(BS) * double(D) * 4.0; // fp32 per element
      double cacheFactor = (bytesPerBlock <= L1Bytes) ? 0.9 : 1.1; // favor cache-fitting blocks
      // Relative cost ~ density * S * D * factor
      double base = double(S) * double(D);
      double cost_bsr = density_bsr * base * cacheFactor;
      double cost_sw = density_sw * base;
      // Env probes: allow disabling one side for HW/impl constraints
      if (llvm::sys::Process::GetEnv("SATTN_DISABLE_BSR")) cost_bsr = std::numeric_limits<double>::infinity();
      if (llvm::sys::Process::GetEnv("SATTN_DISABLE_SW")) cost_sw = std::numeric_limits<double>::infinity();
      // If no window, default to BSR; otherwise pick lower cost
      StringRef spec = (W > 0 && cost_sw < cost_bsr) ? StringRef("sliding_window") : StringRef("bsr");
      op->setAttr("spec", StringAttr::get(ctx, spec));
    });
  }
};
} // namespace

std::unique_ptr<Pass> createSelectSpecPass() { return std::make_unique<SelectSpecPass>(); }

} // namespace mlir::sattn



