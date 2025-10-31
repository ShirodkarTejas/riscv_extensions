add_executable(sattn-opt main.cpp)
target_link_libraries(sattn-opt PRIVATE
  SattnTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIROptLib
  MLIRFuncDialect
)


