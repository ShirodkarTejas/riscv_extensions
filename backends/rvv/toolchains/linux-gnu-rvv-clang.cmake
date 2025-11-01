# Toolchain for cross-compiling RVV backends with Clang (Linux-gnu sysroot)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER clang-18)

# Clang cross target and sysroot, plus RVV flags
set(CMAKE_C_FLAGS_INIT "--target=riscv64-linux-gnu --sysroot=/usr/riscv64-linux-gnu --gcc-toolchain=/usr -O3 -march=rv64gcv_zvl128b -mabi=lp64d -mrvv-vector-bits=128 -menable-experimental-extensions -I/usr/riscv64-linux-gnu/include -L/usr/riscv64-linux-gnu/lib")

set(CMAKE_EXE_LINKER_FLAGS_INIT "--target=riscv64-linux-gnu --gcc-toolchain=/usr -fuse-ld=lld -L/usr/riscv64-linux-gnu/lib -Wl,-rpath-link,/usr/riscv64-linux-gnu/lib -Wl,-dynamic-linker,/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1")


