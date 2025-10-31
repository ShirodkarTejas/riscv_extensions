// Placeholder pass: choose tile sizes and restructure into tiled loops/ops.
// Documentation stub for future MLIR vector/bufferization lowering.

#include "compiler/mlir/transforms/Passes.h"

namespace mlir {
namespace sattn {

class TilePass final /*: public PassWrapper<...>*/ {
public:
  void runOnOperation() {}
};

std::unique_ptr<Pass> createTilePass() { return std::unique_ptr<Pass>(new TilePass()); }

}  // namespace sattn
}  // namespace mlir


