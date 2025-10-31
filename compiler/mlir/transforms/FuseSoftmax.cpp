// Placeholder pass: fuse scale+mask+softmax into a single tile-local op.
// Documentation stub for mapping to softmax_fused primitive.

#include "compiler/mlir/transforms/Passes.h"

namespace mlir {
namespace sattn {

class FuseSoftmaxPass final /*: public PassWrapper<...>*/ {
public:
  void runOnOperation() {}
};

std::unique_ptr<Pass> createFuseSoftmaxPass() {
  return std::unique_ptr<Pass>(new FuseSoftmaxPass());
}

}  // namespace sattn
}  // namespace mlir


