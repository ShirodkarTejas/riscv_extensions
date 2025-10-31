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

  // Program descriptor (use smaller tile for check)
  mmio_write(0x0030, 4);  // m_rows
  mmio_write(0x0038, 16); // head_dim_d
  mmio_write(0x0040, 4);  // block_size
  mmio_write(0x0048, 4);  // k_blocks
  mmio_write(0x0050, 16); // s_tokens
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

  // Compute expected checksum to validate for small tiles
  // Read back params
  auto rd32 = [&](uint64_t addr){ return (uint32_t)mmio_read(addr); };
  uint32_t M = rd32(0x0030), D = rd32(0x0038), BS = rd32(0x0040), S = rd32(0x0050);
  // Load block indices from file if provided, otherwise ramp 0..K-1
  int idx_count = (S + (BS ? BS : 1) - 1) / (BS ? BS : 1);
  int *blk_ids = (int*)malloc(sizeof(int) * (idx_count > 0 ? idx_count : 1));
  int blk_n = 0;
  if (argc > 1) {
    FILE* f = fopen(argv[1], "r");
    if (f) {
      char buf[128];
      while (fgets(buf, sizeof(buf), f) && blk_n < idx_count) {
        int val = atoi(buf);
        blk_ids[blk_n++] = val;
      }
      fclose(f);
    }
  }
  if (blk_n == 0) {
    for (int i = 0; i < idx_count; ++i) blk_ids[i] = i;
    blk_n = idx_count;
  }

  unsigned long long expected = 0ull;
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t t = 0; t < S; ++t) {
      uint32_t block_idx = (BS ? (t / BS) : 0);
      uint32_t block_id = (block_idx < (uint32_t)blk_n) ? (uint32_t)blk_ids[block_idx] : block_idx;
      uint32_t tok_in_block = (BS ? (t % BS) : 0);
      (void)row; // rows are identical; we just multiply by M implicitly via loop
      unsigned long long acc = 0ull;
      for (uint32_t k = 0; k < D; ++k) {
        uint32_t q = ((block_id & 0xFFFFu) << 16) | (k & 0xFFFFu);
        uint32_t kv = (((block_id ^ 0x0f0fu) & 0xFFFFu) << 16) | (k & 0xFFFFu);
        acc += (unsigned long long)q * (unsigned long long)kv;
      }
      expected += acc;
    }
  }
  const char* verdict = (expected == sum_lo) ? "PASS" : "MISMATCH";
  printf("expected=0x%llx -> %s\n", expected, verdict);
  free(blk_ids);
  delete top;
  return 0;
}


