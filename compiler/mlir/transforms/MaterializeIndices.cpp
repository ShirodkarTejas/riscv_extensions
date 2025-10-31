// Materialize sparse indices (BSR) flag on sattn.sparse_attention ops.

#include "compiler/mlir/transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::sattn {

namespace {
struct MaterializeIndicesPass : public PassWrapper<MaterializeIndicesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "sattn.sparse_attention") {
        // Set a boolean unit attribute to indicate indices are materialized
        op->setAttr("materialized_indices", UnitAttr::get(op->getContext()));
        // If indices already exist, do not overwrite
        if (op->getAttr("indices") || op->getAttr("block_indices")) return;
        auto *ctx = op->getContext();
        Builder b(ctx);
        // Read attributes
        auto patternAttr = op->getAttrOfType<StringAttr>("pattern");
        auto tileSAttr = op->getAttrOfType<IntegerAttr>("tile_S");
        auto windowAttr = op->getAttrOfType<IntegerAttr>("window_size");
        auto blockAttr = op->getAttrOfType<IntegerAttr>("block_size");
        auto keepAttr = op->getAttrOfType<FloatAttr>("keep_ratio");
        int64_t S = tileSAttr ? tileSAttr.getInt() : 128;
        SmallVector<Attribute, 8> idxAttrs;
        StringRef pat = patternAttr ? patternAttr.getValue() : StringRef("sliding_global");
        if (pat == "sliding_global") {
          int64_t w = windowAttr ? windowAttr.getInt() : 512;
          int64_t cnt = std::min<int64_t>(S, std::max<int64_t>(1, 2 * w + 1));
          for (int64_t i = 0; i < cnt; ++i) idxAttrs.push_back(b.getI64IntegerAttr(i));
          for (int64_t i = cnt; i < S; ++i) idxAttrs.push_back(b.getI64IntegerAttr(i));
          op->setAttr("indices", b.getArrayAttr(idxAttrs));
          return;
        } else if (pat == "block_topk") {
          int64_t bs = blockAttr ? blockAttr.getInt() : 64;
          double kr = keepAttr ? keepAttr.getValueAsDouble() : 0.12;
          // If a mask is provided, use it to compute block_indices
          if (auto maskList = op->getAttrOfType<ArrayAttr>("mask_blocks")) {
            SmallVector<Attribute, 8> blkAttrs;
            int64_t maxBlocks = std::max<int64_t>(1, (S + bs - 1) / bs);
            for (auto a : maskList) {
              if (auto ia = a.dyn_cast<IntegerAttr>()) blkAttrs.push_back(ia);
              if ((int64_t)blkAttrs.size() >= maxBlocks) break;
            }
            op->setAttr("block_indices", b.getArrayAttr(blkAttrs));
          } else if (auto maskBits = op->getAttrOfType<ArrayAttr>("block_mask")) {
            SmallVector<Attribute, 8> blkAttrs;
            int64_t maxBlocks = std::max<int64_t>(1, (S + bs - 1) / bs);
            for (int64_t i = 0, e = (int64_t)maskBits.size(); i < e; ++i) {
              if (auto ia = maskBits[i].dyn_cast<IntegerAttr>()) {
                if (ia.getInt() != 0) blkAttrs.push_back(b.getI64IntegerAttr(i));
                if ((int64_t)blkAttrs.size() >= maxBlocks) break;
              }
            }
            op->setAttr("block_indices", b.getArrayAttr(blkAttrs));
          } else {
            int64_t kBlocks = std::max<int64_t>(1, (S + bs - 1) / bs * kr);
            SmallVector<Attribute, 8> blkAttrs;
            for (int64_t i = 0; i < kBlocks; ++i) blkAttrs.push_back(b.getI64IntegerAttr(i));
            op->setAttr("block_indices", b.getArrayAttr(blkAttrs));
          }
          return;
        } else {
          for (int64_t i = 0; i < S; ++i) idxAttrs.push_back(b.getI64IntegerAttr(i));
          op->setAttr("indices", b.getArrayAttr(idxAttrs));
          return;
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> createMaterializeIndicesPass() { return std::make_unique<MaterializeIndicesPass>(); }

} // namespace mlir::sattn


