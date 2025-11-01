# Minimal toolchain for cross-compiling RVV backends with Linux-gnu toolchain

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)

# Ensure vector extension and ABI; override as needed via CMAKE_C_FLAGS
set(CMAKE_C_FLAGS_INIT "-O3 -march=rv64gcv -mabi=lp64d")

# Help CMake find target sysroot
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


