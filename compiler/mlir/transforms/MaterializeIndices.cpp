// Placeholder pass: materialize sparse indices (BSR) from pattern attributes.
// This is a documentation stub; real implementation will use MLIR APIs.

#include "compiler/mlir/transforms/Passes.h"

namespace mlir {
namespace sattn {

class MaterializeIndicesPass final /*: public PassWrapper<...>*/ {
public:
  void runOnOperation() {}
};

std::unique_ptr<Pass> createMaterializeIndicesPass() {
  return std::unique_ptr<Pass>(new MaterializeIndicesPass());
}

void registerPasses() {
  // register pass factories when wiring build system
}

}  // namespace sattn
}  // namespace mlir


