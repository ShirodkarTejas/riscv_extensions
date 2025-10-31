cmake_minimum_required(VERSION 3.16)
project(SATTN_MLIR LANGUAGES CXX)

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

add_subdirectory(transforms)
add_subdirectory(tools/sattn-opt)


