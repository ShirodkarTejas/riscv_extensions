#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "verilated.h"
#include "Vrocc_sattn.h"

static inline void step(Vrocc_sattn* top, int n = 1) {
  for (int i = 0; i < n; ++i) {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vrocc_sattn* top = new Vrocc_sattn;
  top->rstn = 0; step(top, 5); top->rstn = 1; step(top, 5);

  auto mmio_write = [&](uint64_t addr, uint64_t data) {
    top->mmio_addr = addr; top->mmio_wdata = data; top->mmio_wen = 1; top->mmio_ren = 0; step(top,1); top->mmio_wen = 0; };
  auto mmio_read  = [&](uint64_t addr) -> uint64_t {
    top->mmio_addr = addr; top->mmio_wen = 0; top->mmio_ren = 1; step(top,1); top->mmio_ren = 0; return top->mmio_rdata; };

  // Optionally read indices from file passed as argv[1]
  if (argc > 1) {
    FILE* f = fopen(argv[1], "r");
    if (f) {
      char buf[128]; int idx = 0;
      while (fgets(buf, sizeof(buf), f)) {
        int val = atoi(buf);
        mmio_write(0x0070, (uint64_t)idx);     // REG_IDX_WADDR
        mmio_write(0x0078, (uint64_t)(val & 0xFFFF));   // REG_IDX_WDATA
        idx++;
        if (idx >= 65536) break;
      }
      fclose(f);
    }
  } else {
    // Populate a few indices in index RAM: write addr then data (commit)
    for (int i = 0; i < 16; ++i) {
      mmio_write(0x0070, (uint64_t)i);     // REG_IDX_WADDR
      mmio_write(0x0078, (uint64_t)(i));   // REG_IDX_WDATA
    }
  }

  // Program minimal descriptor
  mmio_write(0x0030, 64); // m_rows
  mmio_write(0x0038, 64); // head_dim_d
  mmio_write(0x0040, 64); // block_size
  mmio_write(0x0048, 8);  // k_blocks
  mmio_write(0x0050, 512);// s_tokens
  mmio_write(0x0060, 0x14); // CMD_SPDOT_BSR

  // Poll for completion
  int iters = 0;
  while (iters < 100) {
    uint64_t st = mmio_read(0x0060);
    bool done = (st & 1ull) != 0ull;
    bool busy = (st & 2ull) != 0ull;
    if (done) break;
    step(top, 1);
    ++iters;
  }
  uint64_t sum_lo = mmio_read(0x0068);
  printf("verilator_tb: completed in %d iterations, checksum=0x%llx\n", iters, (unsigned long long)sum_lo);
  delete top;
  return 0;
}


