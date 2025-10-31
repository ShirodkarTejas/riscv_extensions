set(SATTN_TRANSFORMS_SRC
  MaterializeIndices.cpp
  Tile.cpp
  FuseSoftmax.cpp
  LowerToRVV.cpp
  LowerToRoCC.cpp
  Vectorize.cpp
  Bufferize.cpp
  Passes.cpp
)

add_mlir_library(SattnTransforms
  ${SATTN_TRANSFORMS_SRC}
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)

target_include_directories(SattnTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)


