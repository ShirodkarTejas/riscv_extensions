#!/usr/bin/env python3
"""
Accuracy Metrics Module for Sparse Attention Benchmarks

Provides comprehensive accuracy/quality metrics beyond simple checksums:
- Mean Absolute Error (MAE)
- Max Absolute Error
- Root Mean Square Error (RMSE)
- Cosine Similarity
- L2 Distance
- Relative Error
- Per-head and per-token breakdown

These metrics help evaluate the numerical accuracy of quantized and
sparse attention implementations compared to fp32 dense baselines.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for attention outputs"""
    
    # Absolute errors
    mae: float                      # Mean Absolute Error
    max_error: float                # Maximum absolute error
    rmse: float                     # Root Mean Square Error
    
    # Relative errors
    relative_mae: float             # MAE / mean(|O_ref|)
    relative_max_error: float       # max_error / max(|O_ref|)
    
    # Similarity metrics
    cosine_similarity: float        # Cosine similarity [-1, 1]
    l2_distance: float              # Euclidean distance
    relative_l2_error: float        # ||O - O_ref|| / ||O_ref||
    
    # Statistical metrics
    pearson_correlation: float      # Pearson correlation coefficient
    
    # Per-dimension breakdown (optional)
    per_head_mae: Optional[np.ndarray] = None
    per_token_mae: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Convert numpy arrays to lists for JSON serialization"""
        if self.per_head_mae is not None and isinstance(self.per_head_mae, np.ndarray):
            self.per_head_mae = self.per_head_mae.tolist()
        if self.per_token_mae is not None and isinstance(self.per_token_mae, np.ndarray):
            self.per_token_mae = self.per_token_mae.tolist()


class AccuracyEvaluator:
    """
    Evaluate accuracy of sparse/quantized attention vs reference implementation.
    
    Usage:
        evaluator = AccuracyEvaluator()
        metrics = evaluator.evaluate(O_test, O_reference)
        print(evaluator.format_report(metrics))
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize accuracy evaluator.
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon
    
    def evaluate(
        self,
        O_test: np.ndarray,
        O_ref: np.ndarray,
        compute_per_head: bool = True,
        compute_per_token: bool = True
    ) -> AccuracyMetrics:
        """
        Compute comprehensive accuracy metrics.
        
        Args:
            O_test: Test output tensor [B, H, L, D]
            O_ref: Reference output tensor [B, H, L, D] (usually fp32 dense)
            compute_per_head: Whether to compute per-head breakdown
            compute_per_token: Whether to compute per-token breakdown
        
        Returns:
            AccuracyMetrics with all computed metrics
        """
        
        # Ensure same shape
        if O_test.shape != O_ref.shape:
            raise ValueError(f"Shape mismatch: {O_test.shape} vs {O_ref.shape}")
        
        # Flatten for some metrics
        O_test_flat = O_test.flatten()
        O_ref_flat = O_ref.flatten()
        
        # 1. Absolute Error Metrics
        abs_error = np.abs(O_test_flat - O_ref_flat)
        mae = float(np.mean(abs_error))
        max_error = float(np.max(abs_error))
        rmse = float(np.sqrt(np.mean(abs_error ** 2)))
        
        # 2. Relative Error Metrics
        ref_mean_abs = np.mean(np.abs(O_ref_flat))
        relative_mae = mae / (ref_mean_abs + self.epsilon)
        
        ref_max_abs = np.max(np.abs(O_ref_flat))
        relative_max_error = max_error / (ref_max_abs + self.epsilon)
        
        # 3. Cosine Similarity
        dot_product = np.dot(O_test_flat, O_ref_flat)
        norm_test = np.linalg.norm(O_test_flat)
        norm_ref = np.linalg.norm(O_ref_flat)
        cosine_sim = float(dot_product / (norm_test * norm_ref + self.epsilon))
        
        # 4. L2 Distance Metrics
        l2_dist = float(np.linalg.norm(O_test_flat - O_ref_flat))
        relative_l2_error = l2_dist / (norm_ref + self.epsilon)
        
        # 5. Pearson Correlation
        if len(O_test_flat) > 1:
            pearson = float(np.corrcoef(O_test_flat, O_ref_flat)[0, 1])
        else:
            pearson = 1.0
        
        # 6. Per-head breakdown (optional)
        per_head_mae_arr = None
        if compute_per_head and O_test.ndim == 4:
            B, H, L, D = O_test.shape
            per_head_mae_arr = np.zeros(H)
            for h in range(H):
                per_head_error = np.abs(O_test[:, h, :, :] - O_ref[:, h, :, :])
                per_head_mae_arr[h] = np.mean(per_head_error)
        
        # 7. Per-token breakdown (optional)
        per_token_mae_arr = None
        if compute_per_token and O_test.ndim == 4:
            B, H, L, D = O_test.shape
            per_token_mae_arr = np.zeros(L)
            for l in range(L):
                per_token_error = np.abs(O_test[:, :, l, :] - O_ref[:, :, l, :])
                per_token_mae_arr[l] = np.mean(per_token_error)
        
        return AccuracyMetrics(
            mae=mae,
            max_error=max_error,
            rmse=rmse,
            relative_mae=relative_mae,
            relative_max_error=relative_max_error,
            cosine_similarity=cosine_sim,
            l2_distance=l2_dist,
            relative_l2_error=relative_l2_error,
            pearson_correlation=pearson,
            per_head_mae=per_head_mae_arr,
            per_token_mae=per_token_mae_arr,
        )
    
    def to_dict(self, metrics: AccuracyMetrics) -> Dict:
        """Convert AccuracyMetrics to dict for JSON serialization"""
        result = asdict(metrics)
        # Convert numpy types to native Python types
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy scalar
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result
    
    def format_report(self, metrics: AccuracyMetrics) -> str:
        """Generate human-readable accuracy report"""
        lines = [
            "=" * 60,
            "Accuracy Metrics Report",
            "=" * 60,
            "",
            "Absolute Error Metrics:",
            f"  Mean Absolute Error (MAE):    {metrics.mae:.6f}",
            f"  Max Absolute Error:           {metrics.max_error:.6f}",
            f"  Root Mean Square Error (RMSE):{metrics.rmse:.6f}",
            "",
            "Relative Error Metrics:",
            f"  Relative MAE:                 {metrics.relative_mae:.6f} ({metrics.relative_mae*100:.3f}%)",
            f"  Relative Max Error:           {metrics.relative_max_error:.6f} ({metrics.relative_max_error*100:.3f}%)",
            f"  Relative L2 Error:            {metrics.relative_l2_error:.6f} ({metrics.relative_l2_error*100:.3f}%)",
            "",
            "Similarity Metrics:",
            f"  Cosine Similarity:            {metrics.cosine_similarity:.6f}",
            f"  Pearson Correlation:          {metrics.pearson_correlation:.6f}",
            f"  L2 Distance:                  {metrics.l2_distance:.6f}",
            "",
        ]
        
        # Per-head breakdown
        if metrics.per_head_mae is not None:
            per_head_list = metrics.per_head_mae if isinstance(metrics.per_head_mae, list) else metrics.per_head_mae.tolist()
            lines.append("Per-Head MAE:")
            for h, mae_h in enumerate(per_head_list):
                lines.append(f"  Head {h:2d}: {mae_h:.6f}")
            lines.append("")
        
        # Per-token breakdown (show first/last 5 if too many)
        if metrics.per_token_mae is not None:
            per_token_list = metrics.per_token_mae if isinstance(metrics.per_token_mae, list) else metrics.per_token_mae.tolist()
            lines.append("Per-Token MAE:")
            if len(per_token_list) <= 10:
                for t, mae_t in enumerate(per_token_list):
                    lines.append(f"  Token {t:3d}: {mae_t:.6f}")
            else:
                # Show first 5
                for t in range(5):
                    lines.append(f"  Token {t:3d}: {per_token_list[t]:.6f}")
                lines.append("  ...")
                # Show last 5
                for t in range(len(per_token_list) - 5, len(per_token_list)):
                    lines.append(f"  Token {t:3d}: {per_token_list[t]:.6f}")
            lines.append("")
        
        # Quality assessment
        lines.append("Quality Assessment:")
        if metrics.relative_mae < 0.01:
            quality = "EXCELLENT (<1% error)"
        elif metrics.relative_mae < 0.05:
            quality = "GOOD (<5% error)"
        elif metrics.relative_mae < 0.10:
            quality = "ACCEPTABLE (<10% error)"
        else:
            quality = "POOR (>10% error)"
        lines.append(f"  Overall Quality: {quality}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def compare_multiple(
        self,
        outputs: Dict[str, np.ndarray],
        reference_key: str = "fp32"
    ) -> Dict[str, AccuracyMetrics]:
        """
        Compare multiple implementations against a reference.
        
        Args:
            outputs: Dict mapping name -> output tensor
            reference_key: Key for reference implementation
        
        Returns:
            Dict mapping name -> AccuracyMetrics
        """
        if reference_key not in outputs:
            raise ValueError(f"Reference '{reference_key}' not found in outputs")
        
        O_ref = outputs[reference_key]
        results = {}
        
        for name, O_test in outputs.items():
            if name == reference_key:
                continue
            results[name] = self.evaluate(O_test, O_ref)
        
        return results
    
    def generate_comparison_table(
        self,
        comparisons: Dict[str, AccuracyMetrics]
    ) -> str:
        """
        Generate markdown table comparing multiple implementations.
        
        Args:
            comparisons: Dict mapping name -> AccuracyMetrics
        
        Returns:
            Markdown table string
        """
        lines = [
            "| Implementation | MAE | Max Error | RMSE | Rel. MAE (%) | Cosine Sim. |",
            "|----------------|-----|-----------|------|--------------|-------------|",
        ]
        
        for name, metrics in sorted(comparisons.items()):
            lines.append(
                f"| {name:14s} | {metrics.mae:.4f} | {metrics.max_error:.4f} | "
                f"{metrics.rmse:.4f} | {metrics.relative_mae*100:.2f} | "
                f"{metrics.cosine_similarity:.6f} |"
            )
        
        return "\n".join(lines)


def main():
    """Demo usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Accuracy metrics demo")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--L", type=int, default=128, help="Sequence length")
    parser.add_argument("--D", type=int, default=32, help="Head dimension")
    parser.add_argument("--noise", type=float, default=0.01, 
                       help="Noise level for test data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Accuracy Metrics Demo")
    print("=" * 60)
    print(f"Problem size: B={args.B}, H={args.H}, L={args.L}, D={args.D}")
    print(f"Simulated noise: {args.noise}")
    print()
    
    # Generate synthetic data
    np.random.seed(42)
    shape = (args.B, args.H, args.L, args.D)
    O_ref = np.random.randn(*shape).astype(np.float32)
    
    # Simulate quantized/sparse output with noise
    O_test = O_ref + np.random.randn(*shape).astype(np.float32) * args.noise
    
    # Evaluate
    evaluator = AccuracyEvaluator()
    metrics = evaluator.evaluate(O_test, O_ref)
    
    print(evaluator.format_report(metrics))
    
    # Multi-implementation comparison demo
    print("\n" + "=" * 60)
    print("Multi-Implementation Comparison Demo")
    print("=" * 60)
    
    outputs = {
        "fp32": O_ref,
        "bf16": O_ref + np.random.randn(*shape).astype(np.float32) * 0.005,
        "i8": O_ref + np.random.randn(*shape).astype(np.float32) * 0.02,
        "i4": O_ref + np.random.randn(*shape).astype(np.float32) * 0.05,
    }
    
    comparisons = evaluator.compare_multiple(outputs, reference_key="fp32")
    print(evaluator.generate_comparison_table(comparisons))
    
    print("\n" + "=" * 60)
    print("JSON Output Example (bf16):")
    print("=" * 60)
    print(json.dumps(evaluator.to_dict(comparisons["bf16"]), indent=2))


if __name__ == "__main__":
    main()

