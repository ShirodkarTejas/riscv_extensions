cmake_minimum_required(VERSION 3.13)
project(sattn_rvv C)

set(CMAKE_C_STANDARD 11)

add_library(sattn_rvv STATIC
  src/sparse_attention_rvv.c
)
target_include_directories(sattn_rvv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sattn_rvv PUBLIC m)

# Example flags for RVV with GCC/Clang (user may override)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d")

add_executable(sattn_rvv_bench bench/measure_cycles.c)
target_link_libraries(sattn_rvv_bench PRIVATE sattn_rvv m)

add_executable(sattn_rvv_bench_blocktopk bench/measure_cycles_blocktopk.c)
target_link_libraries(sattn_rvv_bench_blocktopk PRIVATE sattn_rvv m)

add_executable(sattn_rvv_bench_cli bench/measure_cycles_cli.c)
target_link_libraries(sattn_rvv_bench_cli PRIVATE sattn_rvv m)


