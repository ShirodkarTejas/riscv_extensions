#include "compiler/mlir/transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::sattn::registerPasses();
  return failed(mlir::MlirOptMain(argc, argv, "SATTN optimizer", registry));
}


