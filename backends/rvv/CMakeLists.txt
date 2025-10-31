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

add_executable(sattn_rvv_compare_blocktopk bench/compare_blocktopk_ref.c)
target_link_libraries(sattn_rvv_compare_blocktopk PRIVATE sattn_rvv m)

add_executable(sattn_rvv_test_helpers bench/test_helpers.c)
target_link_libraries(sattn_rvv_test_helpers PRIVATE sattn_rvv m)

add_executable(sattn_rvv_compare_nm bench/compare_nm_ref.c)
target_link_libraries(sattn_rvv_compare_nm PRIVATE sattn_rvv m)

add_executable(sattn_rvv_compare_lsh bench/compare_lsh_ref.c)
target_link_libraries(sattn_rvv_compare_lsh PRIVATE sattn_rvv m)

add_executable(sattn_rvv_compare_sw bench/compare_sw_ref.c)
target_link_libraries(sattn_rvv_compare_sw PRIVATE sattn_rvv m)

add_executable(sattn_rvv_runner bench/rvv_runner.c)
target_link_libraries(sattn_rvv_runner PRIVATE sattn_rvv m)

add_executable(sattn_rvv_compare_sw_tiled bench/compare_sw_tiled_ref.c)
target_link_libraries(sattn_rvv_compare_sw_tiled PRIVATE sattn_rvv m)
add_executable(sattn_rvv_compare_blocktopk_tiled bench/compare_blocktopk_tiled_ref.c)
target_link_libraries(sattn_rvv_compare_blocktopk_tiled PRIVATE sattn_rvv m)
add_executable(sattn_rvv_compare_lsh_tiled bench/compare_lsh_tiled_ref.c)
target_link_libraries(sattn_rvv_compare_lsh_tiled PRIVATE sattn_rvv m)


