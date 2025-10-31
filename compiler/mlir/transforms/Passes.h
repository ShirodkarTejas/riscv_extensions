#ifndef SATTN_COMPILER_MLIR_TRANSFORMS_PASSES_H_
#define SATTN_COMPILER_MLIR_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {
class Pass;

namespace sattn {

void registerPasses();

std::unique_ptr<Pass> createMaterializeIndicesPass();
std::unique_ptr<Pass> createSelectSpecPass();
std::unique_ptr<Pass> createTilePass();
std::unique_ptr<Pass> createFuseSoftmaxPass();
std::unique_ptr<Pass> createLowerToRVVPass();
std::unique_ptr<Pass> createLowerToRoCCPass();
std::unique_ptr<Pass> createVectorizePass();
std::unique_ptr<Pass> createBufferizePass();

}  // namespace sattn
}  // namespace mlir

#endif  // SATTN_COMPILER_MLIR_TRANSFORMS_PASSES_H_


