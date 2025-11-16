#!/usr/bin/env python3
"""
Energy Estimation Module for Sparse Attention Benchmarks

Since we're running on QEMU/simulation, we can't measure real power directly.
This module provides a physics-based energy proxy model using:
1. Compute energy (MAC operations)
2. Memory energy (DRAM/SRAM accesses)
3. Static leakage power

The model is calibrated with industry-standard technology parameters.
"""

import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TechnologyParameters:
    """Technology node parameters for energy estimation"""
    node_nm: int                    # Technology node (e.g., 7, 5, 3)
    voltage_v: float                # Supply voltage (V)
    frequency_ghz: float            # Operating frequency (GHz)
    
    # Compute energy costs (pJ = picojoules)
    mac_fp32_pj: float              # Energy per FP32 MAC
    mac_bf16_pj: float              # Energy per BF16 MAC
    mac_int8_pj: float              # Energy per INT8 MAC
    mac_int4_pj: float              # Energy per INT4 MAC
    
    # Memory energy costs (pJ per byte)
    l1_read_pj_per_byte: float      # L1 cache read
    l1_write_pj_per_byte: float     # L1 cache write
    l2_read_pj_per_byte: float      # L2 cache read (if miss)
    l2_write_pj_per_byte: float     # L2 cache write
    dram_read_pj_per_byte: float    # DRAM read
    dram_write_pj_per_byte: float   # DRAM write
    
    # Static power (mW)
    static_power_mw: float          # Leakage power


# Industry-standard technology parameters (calibrated from literature)
TECH_PARAMS = {
    "7nm": TechnologyParameters(
        node_nm=7,
        voltage_v=0.75,
        frequency_ghz=2.0,
        
        # Compute (from "Energy-Efficient AI" papers)
        mac_fp32_pj=5.0,
        mac_bf16_pj=2.5,
        mac_int8_pj=0.5,
        mac_int4_pj=0.2,
        
        # Memory (from Horowitz 2014, scaled)
        l1_read_pj_per_byte=5.0,
        l1_write_pj_per_byte=6.0,
        l2_read_pj_per_byte=20.0,
        l2_write_pj_per_byte=25.0,
        dram_read_pj_per_byte=640.0,
        dram_write_pj_per_byte=800.0,
        
        # Static (estimated from ARM Cortex)
        static_power_mw=50.0,
    ),
    
    "5nm": TechnologyParameters(
        node_nm=5,
        voltage_v=0.70,
        frequency_ghz=2.5,
        
        # ~40% reduction from 7nm
        mac_fp32_pj=3.0,
        mac_bf16_pj=1.5,
        mac_int8_pj=0.3,
        mac_int4_pj=0.12,
        
        # ~30% reduction from 7nm
        l1_read_pj_per_byte=3.5,
        l1_write_pj_per_byte=4.2,
        l2_read_pj_per_byte=14.0,
        l2_write_pj_per_byte=17.5,
        dram_read_pj_per_byte=640.0,  # DRAM doesn't scale as much
        dram_write_pj_per_byte=800.0,
        
        static_power_mw=35.0,
    ),
    
    "3nm": TechnologyParameters(
        node_nm=3,
        voltage_v=0.65,
        frequency_ghz=3.0,
        
        # ~50% reduction from 5nm
        mac_fp32_pj=1.5,
        mac_bf16_pj=0.75,
        mac_int8_pj=0.15,
        mac_int4_pj=0.06,
        
        # ~40% reduction from 5nm
        l1_read_pj_per_byte=2.1,
        l1_write_pj_per_byte=2.5,
        l2_read_pj_per_byte=8.4,
        l2_write_pj_per_byte=10.5,
        dram_read_pj_per_byte=640.0,
        dram_write_pj_per_byte=800.0,
        
        static_power_mw=25.0,
    ),
}


@dataclass
class EnergyBreakdown:
    """Energy consumption breakdown"""
    compute_energy_uj: float        # Compute energy (microjoules)
    memory_read_energy_uj: float    # Memory read energy
    memory_write_energy_uj: float   # Memory write energy
    static_energy_uj: float         # Static/leakage energy
    total_energy_uj: float          # Total energy
    
    # Derived metrics
    average_power_mw: float         # Average power (mW)
    energy_efficiency_gops_per_w: float  # GOPs/W
    
    # Breakdown percentages
    compute_percent: float
    memory_read_percent: float
    memory_write_percent: float
    static_percent: float


class EnergyEstimator:
    """
    Energy estimation based on performance counters and technology parameters.
    
    Usage:
        estimator = EnergyEstimator(tech_node="7nm")
        energy = estimator.estimate(
            cycles=1000000,
            mac_ops=50000000,
            bytes_read=1048576,
            bytes_written=262144,
            precision="bf16"
        )
    """
    
    def __init__(
        self,
        tech_node: str = "7nm",
        cache_model: str = "optimistic"
    ):
        """
        Initialize energy estimator.
        
        Args:
            tech_node: Technology node ("7nm", "5nm", "3nm")
            cache_model: Cache behavior assumption
                - "optimistic": Assume 80% L1, 15% L2, 5% DRAM
                - "pessimistic": Assume 50% L1, 30% L2, 20% DRAM
                - "worst_case": Assume all DRAM
        """
        if tech_node not in TECH_PARAMS:
            raise ValueError(f"Unknown tech node: {tech_node}. Available: {list(TECH_PARAMS.keys())}")
        
        self.tech = TECH_PARAMS[tech_node]
        self.cache_model = cache_model
        
        # Cache hit rate assumptions
        if cache_model == "optimistic":
            self.l1_hit_rate = 0.80
            self.l2_hit_rate = 0.15
            self.dram_rate = 0.05
        elif cache_model == "pessimistic":
            self.l1_hit_rate = 0.50
            self.l2_hit_rate = 0.30
            self.dram_rate = 0.20
        elif cache_model == "worst_case":
            self.l1_hit_rate = 0.0
            self.l2_hit_rate = 0.0
            self.dram_rate = 1.0
        else:
            raise ValueError(f"Unknown cache model: {cache_model}")
    
    def estimate(
        self,
        cycles: int,
        mac_ops: int,
        bytes_read: int,
        bytes_written: int,
        precision: str = "fp32"
    ) -> EnergyBreakdown:
        """
        Estimate energy consumption from performance counters.
        
        Args:
            cycles: Total cycle count
            mac_ops: Number of MAC operations
            bytes_read: Total bytes read
            bytes_written: Total bytes written
            precision: Data precision ("fp32", "bf16", "i8", "i4")
        
        Returns:
            EnergyBreakdown with detailed energy analysis
        """
        
        # 1. Compute Energy
        mac_energy_pj = self._get_mac_energy(precision)
        compute_energy_pj = mac_ops * mac_energy_pj
        compute_energy_uj = compute_energy_pj / 1e6  # pJ -> uJ
        
        # 2. Memory Read Energy (with cache model)
        mem_read_energy_pj = (
            bytes_read * self.l1_hit_rate * self.tech.l1_read_pj_per_byte +
            bytes_read * self.l2_hit_rate * self.tech.l2_read_pj_per_byte +
            bytes_read * self.dram_rate * self.tech.dram_read_pj_per_byte
        )
        mem_read_energy_uj = mem_read_energy_pj / 1e6
        
        # 3. Memory Write Energy (assume all go to L1 first, then writeback)
        mem_write_energy_pj = (
            bytes_written * self.tech.l1_write_pj_per_byte +
            bytes_written * 0.3 * self.tech.l2_write_pj_per_byte +  # 30% writeback to L2
            bytes_written * 0.1 * self.tech.dram_write_pj_per_byte  # 10% writeback to DRAM
        )
        mem_write_energy_uj = mem_write_energy_pj / 1e6
        
        # 4. Static Energy (leakage)
        time_ms = cycles / (self.tech.frequency_ghz * 1e6)  # cycles / (GHz * 1e6) = ms
        static_energy_uj = self.tech.static_power_mw * time_ms / 1000.0  # mW * ms / 1000 = uJ
        
        # 5. Total Energy
        total_energy_uj = (
            compute_energy_uj +
            mem_read_energy_uj +
            mem_write_energy_uj +
            static_energy_uj
        )
        
        # 6. Derived Metrics
        average_power_mw = total_energy_uj / time_ms if time_ms > 0 else 0.0
        
        # GOPs/W = (mac_ops / 1e9) / (total_energy_uj / 1e6) = mac_ops * 1e3 / total_energy_uj
        gops_per_w = (mac_ops * 1e3 / total_energy_uj) if total_energy_uj > 0 else 0.0
        
        # 7. Percentages
        if total_energy_uj > 0:
            compute_pct = 100.0 * compute_energy_uj / total_energy_uj
            mem_read_pct = 100.0 * mem_read_energy_uj / total_energy_uj
            mem_write_pct = 100.0 * mem_write_energy_uj / total_energy_uj
            static_pct = 100.0 * static_energy_uj / total_energy_uj
        else:
            compute_pct = mem_read_pct = mem_write_pct = static_pct = 0.0
        
        return EnergyBreakdown(
            compute_energy_uj=compute_energy_uj,
            memory_read_energy_uj=mem_read_energy_uj,
            memory_write_energy_uj=mem_write_energy_uj,
            static_energy_uj=static_energy_uj,
            total_energy_uj=total_energy_uj,
            average_power_mw=average_power_mw,
            energy_efficiency_gops_per_w=gops_per_w,
            compute_percent=compute_pct,
            memory_read_percent=mem_read_pct,
            memory_write_percent=mem_write_pct,
            static_percent=static_pct,
        )
    
    def _get_mac_energy(self, precision: str) -> float:
        """Get energy per MAC operation for given precision (in pJ)"""
        precision_lower = precision.lower()
        
        if precision_lower in ["fp32", "f32", "float32"]:
            return self.tech.mac_fp32_pj
        elif precision_lower in ["bf16", "bfloat16"]:
            return self.tech.mac_bf16_pj
        elif precision_lower in ["i8", "int8"]:
            return self.tech.mac_int8_pj
        elif precision_lower in ["i4", "int4"]:
            return self.tech.mac_int4_pj
        else:
            # Default to fp32 for unknown
            print(f"Warning: Unknown precision '{precision}', defaulting to fp32")
            return self.tech.mac_fp32_pj
    
    def to_dict(self, breakdown: EnergyBreakdown) -> Dict:
        """Convert EnergyBreakdown to dict for JSON serialization"""
        return asdict(breakdown)
    
    def format_report(self, breakdown: EnergyBreakdown) -> str:
        """Generate human-readable energy report"""
        lines = [
            "=" * 60,
            f"Energy Estimation Report ({self.tech.node_nm}nm, {self.cache_model} cache)",
            "=" * 60,
            "",
            "Energy Breakdown:",
            f"  Compute:       {breakdown.compute_energy_uj:10.3f} µJ ({breakdown.compute_percent:5.1f}%)",
            f"  Memory Read:   {breakdown.memory_read_energy_uj:10.3f} µJ ({breakdown.memory_read_percent:5.1f}%)",
            f"  Memory Write:  {breakdown.memory_write_energy_uj:10.3f} µJ ({breakdown.memory_write_percent:5.1f}%)",
            f"  Static:        {breakdown.static_energy_uj:10.3f} µJ ({breakdown.static_percent:5.1f}%)",
            f"  " + "-" * 40,
            f"  TOTAL:         {breakdown.total_energy_uj:10.3f} µJ",
            "",
            "Derived Metrics:",
            f"  Average Power: {breakdown.average_power_mw:10.2f} mW",
            f"  Efficiency:    {breakdown.energy_efficiency_gops_per_w:10.2f} GOPs/W",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


def main():
    """Demo usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Energy estimation demo")
    parser.add_argument("--tech", default="7nm", choices=["7nm", "5nm", "3nm"])
    parser.add_argument("--cache", default="optimistic", 
                       choices=["optimistic", "pessimistic", "worst_case"])
    parser.add_argument("--cycles", type=int, default=1000000)
    parser.add_argument("--mac-ops", type=int, default=50000000)
    parser.add_argument("--bytes-read", type=int, default=1048576)
    parser.add_argument("--bytes-written", type=int, default=262144)
    parser.add_argument("--precision", default="bf16", 
                       choices=["fp32", "bf16", "i8", "i4"])
    
    args = parser.parse_args()
    
    estimator = EnergyEstimator(tech_node=args.tech, cache_model=args.cache)
    
    energy = estimator.estimate(
        cycles=args.cycles,
        mac_ops=args.mac_ops,
        bytes_read=args.bytes_read,
        bytes_written=args.bytes_written,
        precision=args.precision
    )
    
    print(estimator.format_report(energy))
    
    print("\nJSON Output:")
    print(json.dumps(estimator.to_dict(energy), indent=2))


if __name__ == "__main__":
    main()

