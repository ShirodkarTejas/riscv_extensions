cmake_minimum_required(VERSION 3.16)
project(SATTN_MLIR LANGUAGES C CXX)

option(SATTN_ENABLE_MLIR "Build SATTN MLIR dialect and passes" ON)

if (NOT SATTN_ENABLE_MLIR)
  message(STATUS "SATTN MLIR disabled via option")
  return()
endif()

find_package(MLIR QUIET CONFIG)
if (NOT MLIR_FOUND)
  message(STATUS "MLIR not found. Skipping MLIR targets.")
  return()
endif()

message(STATUS "Found MLIR: ${MLIR_DIR}")

# Also locate LLVM to import AddLLVM/LLVM CMake functions/macros used by MLIR helpers
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM: ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(transforms)
add_subdirectory(tools/sattn-opt)


# Simple test target to run Python-based MLIR tests
add_custom_target(check-mlir
  COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/compiler/mlir/tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running MLIR Python tests"
)


