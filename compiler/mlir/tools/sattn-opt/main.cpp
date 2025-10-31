#include "transforms/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Register commonly used dialects referenced in tests
  registry.insert<mlir::func::FuncDialect>();
  // Ensure SATTN pipelines TUs are linked and registrations are live
  mlir::sattn::registerPasses();
  // Pipelines are registered in Passes.cpp static initializers
  return failed(mlir::MlirOptMain(argc, argv, "SATTN optimizer", registry));
}


