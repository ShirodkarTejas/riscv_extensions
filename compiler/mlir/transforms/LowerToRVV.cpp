// Placeholder pass: lower tiled ops to RVV-ready vector ops with gather/scatter.
// Documentation stub; real implementation will use MLIR Vector + LLVM lowering.

#include "compiler/mlir/transforms/Passes.h"

namespace mlir {
namespace sattn {

class LowerToRVVPass final /*: public PassWrapper<...>*/ {
public:
  void runOnOperation() {}
};

std::unique_ptr<Pass> createLowerToRVVPass() {
  return std::unique_ptr<Pass>(new LowerToRVVPass());
}

}  // namespace sattn
}  // namespace mlir


