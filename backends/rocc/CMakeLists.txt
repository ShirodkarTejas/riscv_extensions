cmake_minimum_required(VERSION 3.13)
project(sattn_rocc_backend C)

set(CMAKE_C_STANDARD 11)

add_library(sattn_rocc STATIC
  src/sparse_attention_rocc.c
)
target_include_directories(sattn_rocc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})

add_executable(rocc_issue_spdot bench/issue_spdot.c)
target_link_libraries(rocc_issue_spdot PRIVATE sattn_rocc)


