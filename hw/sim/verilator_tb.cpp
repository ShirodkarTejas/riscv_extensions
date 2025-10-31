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

  // Program descriptor (defaults; can be overridden by desc file)
  uint32_t M = 4, D = 16, BS = 4, KB = 4, S = 16, GT = 0, NMN = 0, NMM = 0, LSHB = 0;
  uint32_t GQA_GS = 1, COMP_BS = 0;
  float KEEP = 0.0f;
  // Optionally read a descriptor file passed as argv[2]: key=value per line
  if (argc > 2) {
    FILE* df = fopen(argv[2], "r");
    if (df) {
      char buf[128];
      while (fgets(buf, sizeof(buf), df)) {
        char key[64]; int val;
        if (sscanf(buf, "%63[^=]=%d", key, &val) == 2) {
          if (!strcmp(key, "m_rows")) M = (uint32_t)val;
          else if (!strcmp(key, "head_dim_d")) D = (uint32_t)val;
          else if (!strcmp(key, "block_size")) BS = (uint32_t)val;
          else if (!strcmp(key, "k_blocks")) KB = (uint32_t)val;
          else if (!strcmp(key, "s_tokens")) S = (uint32_t)val;
          else if (!strcmp(key, "global_tokens")) GT = (uint32_t)val;
          else if (!strcmp(key, "nm_n")) NMN = (uint32_t)val;
          else if (!strcmp(key, "nm_m")) NMM = (uint32_t)val;
          else if (!strcmp(key, "lsh_buckets")) LSHB = (uint32_t)val;
        } else if (sscanf(buf, "%63[^=]=%d", key, &val) == 2) {
          // already handled by first branch
        }
      }
      fclose(df);
      // parse keep_ratio if present (float)
      df = fopen(argv[2], "r");
      if (df) {
        char buf2[128];
        while (fgets(buf2, sizeof(buf2), df)) {
          char key2[64]; float fval;
          if (sscanf(buf2, "%63[^=]=%f", key2, &fval) == 2) {
            if (!strcmp(key2, "keep_ratio")) KEEP = fval;
          }
        }
        fclose(df);
      }
      // re-open to parse optional gqa/comp
      df = fopen(argv[2], "r");
      if (df) {
        char buf3[128];
        while (fgets(buf3, sizeof(buf3), df)) {
          char key3[64]; int v3;
          if (sscanf(buf3, "%63[^=]=%d", key3, &v3) == 2) {
            if (!strcmp(key3, "gqa_group_size")) GQA_GS = (uint32_t)v3;
            else if (!strcmp(key3, "comp_block_size")) COMP_BS = (uint32_t)v3;
          }
        }
        fclose(df);
      }
    }
  }
  mmio_write(0x0030, M);
  mmio_write(0x0038, D);
  mmio_write(0x0040, BS);
  mmio_write(0x0048, KB);
  mmio_write(0x0050, S);
  if (GQA_GS) mmio_write(0x00C8, GQA_GS);
  if (COMP_BS) mmio_write(0x00D0, COMP_BS);
  // Program per-spec MMIOs if present
  if (NMN) mmio_write(0x00B8, NMN);
  if (NMM) mmio_write(0x00C0, NMM);
  if (LSHB) mmio_write(0x00C8, LSHB);
  if (KEEP > 0.0f) mmio_write(0x00D0, (uint64_t)(KEEP * 1000.0f));
  mmio_write(0x0060, 0x14); // CMD_SPDOT_BSR

  // Poll for completion
  int iters = 0;
  while (iters < 10000) {
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
  M = rd32(0x0030); D = rd32(0x0038); BS = rd32(0x0040); S = rd32(0x0050);
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

  // Issue softmax_fused (0x15) and validate stub checksum: sum of (i_row + s_tok) over M x S
  mmio_write(0x0060, 0x15); // CMD_SOFTMAX_FUS
  iters = 0; while (iters < 10000) { uint64_t st = mmio_read(0x0060); if (st & 1ull) break; step(top,1); ++iters; }
  uint64_t sof_sum = mmio_read(0x0080);
  unsigned long long sof_expected = 0ull;
  for (uint32_t row = 0; row < M; ++row) for (uint32_t t = 0; t < S; ++t) sof_expected += (unsigned long long)(row + t);
  printf("softmax_fused checksum=0x%llx expected=0x%llx -> %s\n",
         (unsigned long long)sof_sum, sof_expected, (sof_sum == sof_expected ? "PASS" : "MISMATCH"));

  // Issue spmm_bsr (0x16) and validate stub checksum: sum of (i_row + s_tok + d_dim) over M x S x D
  mmio_write(0x0060, 0x16); // CMD_SPMM_BSR
  iters = 0; while (iters < 10000) { uint64_t st = mmio_read(0x0060); if (st & 1ull) break; step(top,1); ++iters; }
  uint64_t spm_sum = mmio_read(0x0088);
  unsigned long long spm_expected = 0ull;
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t t = 0; t < S; ++t)
      for (uint32_t k = 0; k < D; ++k)
        spm_expected += (unsigned long long)(row + t + k);
  printf("spmm_bsr checksum=0x%llx expected=0x%llx -> %s\n",
         (unsigned long long)spm_sum, spm_expected, (spm_sum == spm_expected ? "PASS" : "MISMATCH"));
  // Read RTL counters and also compute proxies for comparison
  uint64_t gcy = mmio_read(0x0090);
  uint64_t mcy = mmio_read(0x0098);
  uint64_t dma = mmio_read(0x00A0);
  uint64_t dmaq = mmio_read(0x00A8);
  uint64_t dmak = mmio_read(0x00B0);
  uint64_t proxy_gcy = (uint64_t)S * (uint64_t)D;
  uint64_t proxy_mcy = (uint64_t)M * (uint64_t)S * (uint64_t)D;
  uint64_t proxy_dma = ((uint64_t)S * (uint64_t)D) * 8ull; // Q+K 4B each per element
  // BLG overhead: global tokens contribute extra Q/K traffic and gather effort
  if (GT > 0) {
    proxy_gcy += (uint64_t)GT * (uint64_t)D;
    proxy_dma += (uint64_t)GT * (uint64_t)D * 8ull;
  }
  printf("rocc_counters(rtl):   gather_cycles=%llu mac_cycles=%llu dma_bytes=%llu dma_q=%llu dma_k=%llu\n",
         (unsigned long long)gcy, (unsigned long long)mcy, (unsigned long long)dma,
         (unsigned long long)dmaq, (unsigned long long)dmak);
  printf("rocc_counters(proxy): gather_cycles=%llu mac_cycles=%llu dma_bytes=%llu\n",
         (unsigned long long)proxy_gcy, (unsigned long long)proxy_mcy, (unsigned long long)proxy_dma);
  printf("spec_info: global_tokens=%u gqa_group_size=%u comp_block_size=%u nm=(%u,%u) lsh_buckets=%u keep_ratio=%.3f\n", GT, GQA_GS, COMP_BS, NMN, NMM, LSHB, KEEP);
  // Print simple utilization estimates
  unsigned long long expected_mac = (unsigned long long)M * (unsigned long long)S * (unsigned long long)D;
  double util_mac = expected_mac ? ((double)mcy / (double)expected_mac) : 0.0;
  double util_gather = ((double)gcy) / ((double)((unsigned long long)S * (unsigned long long)D) + 1e-9);
  printf("rocc_util: util_mac=%.3f util_gather=%.3f expected_mac=%llu\n", util_mac, util_gather, expected_mac);
  delete top;
  return 0;
}


