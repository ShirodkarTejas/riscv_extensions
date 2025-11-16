#!/usr/bin/env python3
"""
Variant Selector Tool

Automatically selects the best sparse attention pattern and precision
based on user-defined constraints (power budget, latency target, memory limit).

Usage:
    # Select by constraints
    python bench/variant_selector.py \\
        --max-memory-mb 1.0 \\
        --max-latency-ms 30 \\
        --max-energy-uj 200 \\
        --L 128 --D 32

    # Or find Pareto-optimal configurations
    python bench/variant_selector.py \\
        --pareto \\
        --objectives latency,energy,memory \\
        --L 128 --D 32
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@dataclass
class Constraint:
    """User-defined constraint"""
    metric: str
    operator: str  # 'le' (<=), 'ge' (>=), 'eq' (==)
    value: float


@dataclass
class Configuration:
    """A specific pattern + precision configuration"""
    pattern: str
    precision: str
    metrics: Dict[str, float]
    
    def satisfies_constraint(self, constraint: Constraint) -> bool:
        """Check if this configuration satisfies a constraint"""
        value = self.get_metric(constraint.metric)
        if value is None:
            return False
        
        if constraint.operator == 'le':
            return value <= constraint.value
        elif constraint.operator == 'ge':
            return value >= constraint.value
        elif constraint.operator == 'eq':
            return abs(value - constraint.value) < 1e-6
        return False
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get a metric value, handling nested dicts"""
        # Handle nested metrics like 'performance.latency_ms'
        if '.' in metric_name:
            parts = metric_name.split('.')
            value = self.metrics
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return float(value) if value is not None else None
        return self.metrics.get(metric_name)
    
    def __str__(self):
        return f"{self.pattern} ({self.precision})"


class VariantSelector:
    """
    Intelligent variant selection based on constraints and objectives.
    
    Can either:
    1. Find configurations that satisfy hard constraints
    2. Compute Pareto-optimal frontiers for multi-objective optimization
    """
    
    def __init__(self, results_dir: str = "bench/results"):
        self.results_dir = results_dir
        self.configurations: List[Configuration] = []
    
    def load_benchmark_results(self, results_file: str = "UNIFIED_PATTERN_COMPARISON.json"):
        """Load benchmark results from JSON file"""
        results_path = os.path.join(self.results_dir, results_file)
        
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found. Using example data.")
            self._load_example_data()
            return
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Parse configurations from results
        for result in data.get('results', []):
            config = Configuration(
                pattern=result['pattern'],
                precision=result['precision'],
                metrics=result['metrics']
            )
            self.configurations.append(config)
    
    def _load_example_data(self):
        """Load example data for demonstration"""
        # Simulate data based on our comprehensive benchmarks
        example_configs = [
            # sliding_window
            {"pattern": "sliding_window", "precision": "i4", "memory_mb": 0.16, "latency_ms": 25, "energy_uj": 150},
            {"pattern": "sliding_window", "precision": "i8", "memory_mb": 0.32, "latency_ms": 23, "energy_uj": 180},
            {"pattern": "sliding_window", "precision": "bf16", "memory_mb": 1.28, "latency_ms": 19, "energy_uj": 250},
            {"pattern": "sliding_window", "precision": "fp32", "memory_mb": 2.56, "latency_ms": 20, "energy_uj": 400},
            # block_local_global
            {"pattern": "block_local_global", "precision": "i4", "memory_mb": 0.30, "latency_ms": 30, "energy_uj": 200},
            {"pattern": "block_local_global", "precision": "i8", "memory_mb": 0.60, "latency_ms": 28, "energy_uj": 240},
            {"pattern": "block_local_global", "precision": "bf16", "memory_mb": 2.40, "latency_ms": 22, "energy_uj": 320},
            {"pattern": "block_local_global", "precision": "fp32", "memory_mb": 4.80, "latency_ms": 24, "energy_uj": 500},
            # nm_structured
            {"pattern": "nm_structured", "precision": "i4", "memory_mb": 0.44, "latency_ms": 35, "energy_uj": 220},
            {"pattern": "nm_structured", "precision": "i8", "memory_mb": 0.88, "latency_ms": 32, "energy_uj": 270},
            {"pattern": "nm_structured", "precision": "bf16", "memory_mb": 3.50, "latency_ms": 26, "energy_uj": 350},
            {"pattern": "nm_structured", "precision": "fp32", "memory_mb": 7.00, "latency_ms": 28, "energy_uj": 550},
            # lsh
            {"pattern": "lsh", "precision": "i4", "memory_mb": 0.32, "latency_ms": 28, "energy_uj": 180},
            {"pattern": "lsh", "precision": "i8", "memory_mb": 0.64, "latency_ms": 26, "energy_uj": 220},
            {"pattern": "lsh", "precision": "bf16", "memory_mb": 2.56, "latency_ms": 21, "energy_uj": 300},
            {"pattern": "lsh", "precision": "fp32", "memory_mb": 5.12, "latency_ms": 23, "energy_uj": 480},
            # landmark
            {"pattern": "landmark", "precision": "i4", "memory_mb": 0.25, "latency_ms": 24, "energy_uj": 160},
            {"pattern": "landmark", "precision": "i8", "memory_mb": 0.50, "latency_ms": 22, "energy_uj": 190},
            {"pattern": "landmark", "precision": "bf16", "memory_mb": 2.00, "latency_ms": 18, "energy_uj": 260},
            {"pattern": "landmark", "precision": "fp32", "memory_mb": 4.00, "latency_ms": 19, "energy_uj": 420},
        ]
        
        for cfg in example_configs:
            metrics = {
                "memory_mb": cfg["memory_mb"],
                "performance": {"latency_ms": cfg["latency_ms"]},
                "energy": {"total_uj": cfg["energy_uj"]},
            }
            config = Configuration(
                pattern=cfg["pattern"],
                precision=cfg["precision"],
                metrics=metrics
            )
            self.configurations.append(config)
    
    def filter_by_constraints(self, constraints: List[Constraint]) -> List[Configuration]:
        """Find all configurations that satisfy given constraints"""
        valid_configs = []
        
        for config in self.configurations:
            satisfies_all = all(config.satisfies_constraint(c) for c in constraints)
            if satisfies_all:
                valid_configs.append(config)
        
        return valid_configs
    
    def find_pareto_optimal(self, objectives: List[str], maximize: List[bool] = None) -> List[Configuration]:
        """
        Find Pareto-optimal configurations for given objectives.
        
        Args:
            objectives: List of metric names to optimize
            maximize: List of booleans indicating whether to maximize each objective
                     (default: all False, i.e., minimize)
        
        Returns:
            List of Pareto-optimal configurations
        """
        if maximize is None:
            maximize = [False] * len(objectives)
        
        # Extract objective values for all configs
        valid_configs = []
        for config in self.configurations:
            values = [config.get_metric(obj) for obj in objectives]
            if all(v is not None for v in values):
                valid_configs.append((config, values))
        
        # Find Pareto frontier
        pareto_optimal = []
        
        for i, (config_i, values_i) in enumerate(valid_configs):
            is_dominated = False
            
            for j, (config_j, values_j) in enumerate(valid_configs):
                if i == j:
                    continue
                
                # Check if config_j dominates config_i
                better_in_all = True
                strictly_better_in_at_least_one = False
                
                for k, (vi, vj) in enumerate(zip(values_i, values_j)):
                    if maximize[k]:
                        if vj <= vi:
                            better_in_all = False
                        if vj > vi:
                            strictly_better_in_at_least_one = True
                    else:  # minimize
                        if vj >= vi:
                            better_in_all = False
                        if vj < vi:
                            strictly_better_in_at_least_one = True
                
                if better_in_all and strictly_better_in_at_least_one:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(config_i)
        
        return pareto_optimal
    
    def rank_by_objective(self, objective: str, maximize: bool = False) -> List[Tuple[Configuration, float]]:
        """Rank all configurations by a single objective"""
        ranked = []
        
        for config in self.configurations:
            value = config.get_metric(objective)
            if value is not None:
                ranked.append((config, value))
        
        ranked.sort(key=lambda x: x[1], reverse=maximize)
        return ranked
    
    def generate_recommendation_report(
        self, 
        constraints: Optional[List[Constraint]] = None,
        pareto_objectives: Optional[List[str]] = None
    ) -> str:
        """Generate a human-readable recommendation report"""
        lines = [
            "=" * 80,
            "Variant Selection Report",
            "=" * 80,
            "",
        ]
        
        if constraints:
            lines.append("### Constraint-Based Selection ###")
            lines.append("")
            lines.append("Constraints:")
            for c in constraints:
                op_str = {"le": "<=", "ge": ">=", "eq": "=="}[c.operator]
                lines.append(f"  - {c.metric} {op_str} {c.value}")
            lines.append("")
            
            valid = self.filter_by_constraints(constraints)
            lines.append(f"Configurations satisfying constraints: {len(valid)}")
            lines.append("")
            
            if valid:
                lines.append("| Pattern | Precision | Memory (MB) | Latency (ms) | Energy (µJ) |")
                lines.append("|---------|-----------|-------------|--------------|-------------|")
                for config in valid:
                    mem = config.get_metric("memory_mb") or 0
                    lat = config.get_metric("performance.latency_ms") or 0
                    eng = config.get_metric("energy.total_uj") or 0
                    lines.append(
                        f"| {config.pattern:18s} | {config.precision:9s} | "
                        f"{mem:11.2f} | {lat:12.2f} | {eng:11.2f} |"
                    )
            else:
                lines.append("❌ No configurations satisfy all constraints!")
                lines.append("")
                lines.append("**Suggestion**: Relax one or more constraints.")
            
            lines.append("")
        
        if pareto_objectives:
            lines.append("### Pareto-Optimal Configurations ###")
            lines.append("")
            lines.append(f"Objectives: {', '.join(pareto_objectives)}")
            lines.append("")
            
            pareto = self.find_pareto_optimal(pareto_objectives)
            lines.append(f"Pareto-optimal configurations: {len(pareto)}")
            lines.append("")
            
            lines.append("| Pattern | Precision | Memory (MB) | Latency (ms) | Energy (µJ) |")
            lines.append("|---------|-----------|-------------|--------------|-------------|")
            for config in pareto:
                mem = config.get_metric("memory_mb") or 0
                lat = config.get_metric("performance.latency_ms") or 0
                eng = config.get_metric("energy.total_uj") or 0
                lines.append(
                    f"| {config.pattern:18s} | {config.precision:9s} | "
                    f"{mem:11.2f} | {lat:12.2f} | {eng:11.2f} |"
                )
            lines.append("")
        
        # Add top recommendations by key metrics
        lines.append("### Top Recommendations by Metric ###")
        lines.append("")
        
        metrics_to_rank = [
            ("memory_mb", "Lowest Memory", False),
            ("performance.latency_ms", "Fastest Latency", False),
            ("energy.total_uj", "Lowest Energy", False),
        ]
        
        for metric, label, maximize in metrics_to_rank:
            ranked = self.rank_by_objective(metric, maximize)
            if ranked:
                config, value = ranked[0]
                lines.append(f"**{label}**: {config} ({value:.2f})")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Variant selector for sparse attention')
    
    # Constraint-based selection
    parser.add_argument('--max-memory-mb', type=float, help='Maximum memory in MB')
    parser.add_argument('--max-latency-ms', type=float, help='Maximum latency in ms')
    parser.add_argument('--max-energy-uj', type=float, help='Maximum energy in µJ')
    parser.add_argument('--min-efficiency', type=float, help='Minimum efficiency (GOPs/W)')
    
    # Pareto optimization
    parser.add_argument('--pareto', action='store_true', 
                       help='Compute Pareto-optimal frontier')
    parser.add_argument('--objectives', default='memory_mb,performance.latency_ms,energy.total_uj',
                       help='Comma-separated list of objectives for Pareto optimization')
    
    # Problem size (for context)
    parser.add_argument('--L', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=32, help='Head dimension')
    
    # Output
    parser.add_argument('--output', help='Output file for recommendations')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Sparse Attention Variant Selector")
    print("=" * 80)
    print(f"Problem size: L={args.L}, D={args.D}")
    print()
    
    # Load configurations
    selector = VariantSelector()
    selector.load_benchmark_results()
    print(f"Loaded {len(selector.configurations)} configurations")
    print()
    
    # Build constraints
    constraints = []
    if args.max_memory_mb:
        constraints.append(Constraint("memory_mb", "le", args.max_memory_mb))
    if args.max_latency_ms:
        constraints.append(Constraint("performance.latency_ms", "le", args.max_latency_ms))
    if args.max_energy_uj:
        constraints.append(Constraint("energy.total_uj", "le", args.max_energy_uj))
    if args.min_efficiency:
        constraints.append(Constraint("energy.efficiency_gops_per_w", "ge", args.min_efficiency))
    
    # Pareto objectives
    pareto_objectives = None
    if args.pareto:
        pareto_objectives = args.objectives.split(',')
    
    # Generate report
    report = selector.generate_recommendation_report(
        constraints=constraints if constraints else None,
        pareto_objectives=pareto_objectives
    )
    
    print(report)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()

