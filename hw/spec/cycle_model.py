import math
from dataclasses import dataclass


@dataclass
class HWParams:
    macs_per_cycle: float = 1024.0
    exp_throughput: float = 8.0  # exps per cycle
    sram_bw_bytes_per_cycle: float = 128.0
    bytes_per_elem: int = 4
    overlap: float = 0.5  # fraction of gather overlapped with compute


def cycles_spdot_bsr(M: int, D: int, S: int, hw: HWParams) -> float:
    compute = (M * D * S) / hw.macs_per_cycle
    bytes_moved = (D * S + M * D) * hw.bytes_per_elem
    gather = bytes_moved / hw.sram_bw_bytes_per_cycle
    return compute + (1 - hw.overlap) * gather


def cycles_softmax_fused(M: int, S: int, hw: HWParams) -> float:
    red = M * math.ceil(math.log2(max(1, S)))
    expn = (M * S) / hw.exp_throughput
    return red + expn


def cycles_spmm_bsr(M: int, S: int, D: int, hw: HWParams) -> float:
    return (M * S * D) / hw.macs_per_cycle


def estimate_total(M: int, D: int, block_size: int, k_blocks: int, hw: HWParams) -> float:
    S = block_size * k_blocks
    return cycles_spdot_bsr(M, D, S, hw) + cycles_softmax_fused(M, S, hw) + cycles_spmm_bsr(M, S, D, hw)


if __name__ == "__main__":
    hw = HWParams()
    print("cycles_total", estimate_total(M=64, D=64, block_size=64, k_blocks=8, hw=hw))


