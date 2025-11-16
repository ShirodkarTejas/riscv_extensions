"""
Dense attention reference implementation for accuracy validation.

This module provides ground-truth dense attention implementations in fp32
for validating sparse and quantized versions.
"""

import numpy as np
from typing import Optional


def dense_attention_fp32(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: Optional[float] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reference dense attention implementation in fp32.
    
    This is the ground truth implementation used for validating sparse
    and quantized attention variants.
    
    Args:
        Q: Query tensor [B, H, L, D] or [L, D]
        K: Key tensor [B, H, L, D] or [L, D]
        V: Value tensor [B, H, L, D] or [L, D]
        scale: Attention scale factor (default: 1/sqrt(D))
        mask: Optional attention mask [L, L]
    
    Returns:
        O: Output tensor, same shape as inputs
    
    Algorithm:
        1. scores = Q @ K^T / sqrt(D)
        2. weights = softmax(scores, dim=-1)
        3. O = weights @ V
    """
    # Handle both batched and unbatched inputs
    if Q.ndim == 2:
        Q = Q[None, None, :, :]
        K = K[None, None, :, :]
        V = V[None, None, :, :]
        was_unbatched = True
    else:
        was_unbatched = False
    
    B, H, L, D = Q.shape
    
    # Default scale
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    
    # Allocate output
    O = np.zeros_like(Q, dtype=np.float32)
    
    # Process each batch and head
    for b in range(B):
        for h in range(H):
            # Compute attention scores: Q @ K^T
            scores = Q[b, h] @ K[b, h].T  # [L, L]
            
            # Scale
            scores = scores * scale
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask
            
            # Softmax over keys (dim=-1)
            scores_max = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)
            
            # Compute output: weights @ V
            O[b, h] = weights @ V[b, h]  # [L, D]
    
    if was_unbatched:
        return O[0, 0]
    
    return O


def dense_attention_bf16(Q, K, V, scale=None, mask=None):
    """Dense attention with bf16 simulation (using fp32 compute)"""
    # Note: NumPy doesn't have native bf16, so we simulate with fp32
    # In real deployment, use proper bf16 implementation
    return dense_attention_fp32(Q, K, V, scale, mask)


def compare_sparse_vs_dense(
    sparse_output: np.ndarray,
    dense_output: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Compare sparse attention output with dense reference.
    
    Args:
        sparse_output: Output from sparse attention [B, H, L, D]
        dense_output: Output from dense reference [B, H, L, D]
        verbose: Print detailed statistics
    
    Returns:
        metrics: Dictionary with accuracy metrics
            - mae: Mean Absolute Error
            - rmse: Root Mean Square Error
            - max_error: Maximum absolute error
            - rel_error: Relative error
            - cosine_similarity: Cosine similarity
    """
    # Flatten for easier computation
    sparse_flat = sparse_output.flatten()
    dense_flat = dense_output.flatten()
    
    # Compute metrics
    diff = sparse_flat - dense_flat
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))
    
    # Relative error
    dense_norm = np.linalg.norm(dense_flat)
    if dense_norm > 0:
        rel_error = np.linalg.norm(diff) / dense_norm
    else:
        rel_error = 0.0
    
    # Cosine similarity
    sparse_norm = np.linalg.norm(sparse_flat)
    if sparse_norm > 0 and dense_norm > 0:
        cosine_sim = np.dot(sparse_flat, dense_flat) / (sparse_norm * dense_norm)
    else:
        cosine_sim = 1.0
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "max_error": float(max_error),
        "rel_error": float(rel_error),
        "cosine_similarity": float(cosine_sim),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ACCURACY METRICS (Sparse vs Dense)")
        print("="*60)
        print(f"Mean Absolute Error (MAE):     {mae:.6f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
        print(f"Maximum Error:                 {max_error:.6f}")
        print(f"Relative Error:                {rel_error:.6f}")
        print(f"Cosine Similarity:             {cosine_sim:.6f}")
        print("="*60)
    
    return metrics


def validate_attention_correctness(
    sparse_fn,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    tolerance: dict = None,
    verbose: bool = True
) -> bool:
    """
    Validate that sparse attention matches dense reference within tolerance.
    
    Args:
        sparse_fn: Sparse attention function to test
        Q, K, V: Input tensors
        tolerance: Dict with tolerance thresholds:
            - mae_threshold: Maximum allowed MAE
            - cosine_threshold: Minimum allowed cosine similarity
        verbose: Print results
    
    Returns:
        passed: True if all checks pass
    """
    if tolerance is None:
        tolerance = {
            "mae_threshold": 0.1,
            "cosine_threshold": 0.95,
        }
    
    # Compute dense reference
    dense_output = dense_attention_fp32(Q, K, V)
    
    # Compute sparse output
    sparse_output = sparse_fn(Q, K, V)
    
    # Compare
    metrics = compare_sparse_vs_dense(sparse_output, dense_output, verbose=verbose)
    
    # Check thresholds
    passed = True
    if metrics["mae"] > tolerance["mae_threshold"]:
        if verbose:
            print(f"❌ MAE too high: {metrics['mae']:.6f} > {tolerance['mae_threshold']:.6f}")
        passed = False
    
    if metrics["cosine_similarity"] < tolerance["cosine_threshold"]:
        if verbose:
            print(f"❌ Cosine similarity too low: {metrics['cosine_similarity']:.6f} < {tolerance['cosine_threshold']:.6f}")
        passed = False
    
    if passed and verbose:
        print("✅ All accuracy checks passed!")
    
    return passed


if __name__ == "__main__":
    print("Testing dense attention reference implementation...\n")
    
    # Test with small example
    np.random.seed(42)
    L, D = 8, 4
    Q = np.random.randn(L, D).astype(np.float32)
    K = np.random.randn(L, D).astype(np.float32)
    V = np.random.randn(L, D).astype(np.float32)
    
    # Compute attention
    O = dense_attention_fp32(Q, K, V)
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Output shape: O={O.shape}")
    print(f"Output range: [{O.min():.4f}, {O.max():.4f}]")
    print(f"Output mean: {O.mean():.4f}")
    
    # Test batched version
    print("\nTesting batched version...")
    B, H, L, D = 2, 4, 16, 8
    Q = np.random.randn(B, H, L, D).astype(np.float32)
    K = np.random.randn(B, H, L, D).astype(np.float32)
    V = np.random.randn(B, H, L, D).astype(np.float32)
    
    O = dense_attention_fp32(Q, K, V)
    print(f"Batched output shape: {O.shape}")
    
    # Test self-consistency
    print("\nTesting self-consistency...")
    O1 = dense_attention_fp32(Q, K, V)
    O2 = dense_attention_fp32(Q, K, V)
    diff = np.max(np.abs(O1 - O2))
    print(f"Max difference between two runs: {diff}")
    assert diff < 1e-6, "Dense attention is not deterministic!"
    
    print("\n✅ All tests passed!")

