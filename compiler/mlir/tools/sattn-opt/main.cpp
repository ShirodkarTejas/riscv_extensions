#include "transforms/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Minimal registry; custom passes optional
  return failed(mlir::MlirOptMain(argc, argv, "SATTN optimizer", registry));
}


