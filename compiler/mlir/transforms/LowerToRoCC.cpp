// Placeholder pass: lower tiled ops to custom RoCC primitives calls.
// Documentation stub for mapping to spdot_bsr, softmax_fused, spmm_bsr.

#include "compiler/mlir/transforms/Passes.h"

namespace mlir {
namespace sattn {

class LowerToRoCCPass final /*: public PassWrapper<...>*/ {
public:
  void runOnOperation() {}
};

std::unique_ptr<Pass> createLowerToRoCCPass() {
  return std::unique_ptr<Pass>(new LowerToRoCCPass());
}

}  // namespace sattn
}  // namespace mlir


