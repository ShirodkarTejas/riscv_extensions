#!/usr/bin/env python3
"""
PyTorch to RISC-V RVV Compiler (Phase 2)

Compiles PyTorch sparse attention models to optimized RISC-V C code.

Pipeline:
  PyTorch Model → MLIR (Torch dialect) → MLIR (SATTN dialect) → C Code

Usage:
    python compile_from_pytorch.py \
        --pattern sliding_window \
        --precision i8 \
        --output generated_attention.c

Example:
    from python.torch_sparse_attention import SparseAttentionLayer
    
    model = SparseAttentionLayer(
        dim=64, num_heads=8,
        pattern="sliding_window",
        precision="i8",
        window_size=16
    )
    
    # Export to MLIR and generate C code
    generate_c_code(model, "output.c")
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# MLIR Generation (Placeholder - would use torch-mlir in practice)
# ===================================================================

def generate_mlir_from_pytorch(
    pattern: str,
    precision: str,
    B: int = 1,
    H: int = 8,
    L: int = 128,
    D: int = 64,
    **kwargs
) -> str:
    """
    Generate MLIR IR for a sparse attention operation.
    
    In a full implementation, this would:
    1. Use torch-mlir to export PyTorch model
    2. Apply pattern matching to identify sparse attention
    3. Lower to sattn dialect
    
    For now, we generate the SATTN dialect directly.
    """
    
    # Get pattern-specific attributes
    attrs = get_pattern_attributes(pattern, precision, **kwargs)
    
    # Generate MLIR module
    mlir_code = f"""
// Generated MLIR for {pattern} with {precision} precision
module {{
  func.func @sparse_attention_main(%q: tensor<{B}x{H}x{L}x{D}xf32>,
                                    %k: tensor<{B}x{H}x{L}x{D}xf32>,
                                    %v: tensor<{B}x{H}x{L}x{D}xf32>) -> tensor<{B}x{H}x{L}x{D}xf32> {{
    %output = sattn.{get_op_name(pattern)}(%q, %k, %v) {{
"""
    
    # Add pattern-specific attributes
    for key, value in attrs.items():
        if isinstance(value, int):
            mlir_code += f"      {key} = {value} : i64,\n"
        elif isinstance(value, float):
            mlir_code += f"      {key} = {value} : f32,\n"
        elif isinstance(value, str):
            mlir_code += f'      {key} = "{value}",\n'
    
    mlir_code += f"""    }} : (tensor<{B}x{H}x{L}x{D}xf32>, tensor<{B}x{H}x{L}x{D}xf32>, 
         tensor<{B}x{H}x{L}x{D}xf32>) -> tensor<{B}x{H}x{L}x{D}xf32>
    return %output : tensor<{B}x{H}x{L}x{D}xf32>
  }}
}}
"""
    
    return mlir_code


def get_op_name(pattern: str) -> str:
    """Map pattern name to MLIR op name"""
    mapping = {
        "sliding_window": "sliding_window",
        "block_local_global": "block_local_global",
        "nm_structured": "nm_structured",
        "lsh": "lsh",
        "landmark": "landmark",
    }
    return mapping.get(pattern, "sparse_attention")


def get_pattern_attributes(pattern: str, precision: str, **kwargs) -> Dict:
    """Get pattern-specific MLIR attributes"""
    attrs = {"precision": precision}
    
    if pattern == "sliding_window":
        attrs["window_size"] = kwargs.get("window_size", 16)
    elif pattern == "block_local_global":
        attrs["block_size"] = kwargs.get("block_size", 16)
        attrs["keep_ratio"] = kwargs.get("keep_ratio", 0.10)
        attrs["global_tokens"] = kwargs.get("global_tokens", 4)
    elif pattern == "nm_structured":
        attrs["nm_n"] = kwargs.get("nm_n", 2)
        attrs["nm_m"] = kwargs.get("nm_m", 4)
    elif pattern == "lsh":
        attrs["buckets"] = kwargs.get("buckets", 8)
    elif pattern == "landmark":
        attrs["num_landmarks"] = kwargs.get("num_landmarks", 16)
    
    # Add quantization scales for i8/i4
    if precision in ["i8", "i4"]:
        attrs["scale_q"] = kwargs.get("scale_q", 0.020 if precision == "i8" else 0.008)
        attrs["scale_k"] = kwargs.get("scale_k", 0.020 if precision == "i8" else 0.008)
        attrs["scale_v"] = kwargs.get("scale_v", 0.020 if precision == "i8" else 0.008)
    
    return attrs


# ===================================================================
# C Code Generation
# ===================================================================

def generate_c_code(
    pattern: str,
    precision: str,
    B: int = 1,
    H: int = 8,
    L: int = 128,
    D: int = 64,
    **kwargs
) -> str:
    """
    Generate C code that calls our validated RVV backend.
    
    This directly generates C code calling the functions we validated in Phase 1.
    """
    
    # Get function name
    func_name = get_rvv_function_name(pattern, precision)
    
    # Generate struct type
    params_struct = get_params_struct_name(pattern)
    
    # Generate initialization
    params_init = get_params_initialization(pattern, **kwargs)
    
    # Generate scales for quantization
    scales = ""
    if precision in ["i8", "i4"]:
        scale_q = kwargs.get("scale_q", 0.020 if precision == "i8" else 0.008)
        scale_k = kwargs.get("scale_k", 0.020 if precision == "i8" else 0.008)
        scale_v = kwargs.get("scale_v", 0.020 if precision == "i8" else 0.008)
        scales = f", {scale_q}f, {scale_k}f, {scale_v}f"
    
    # Generate C code
    c_code = f"""// Generated from MLIR - Phase 2
// Pattern: {pattern}, Precision: {precision}
// This code calls the validated RVV backend from Phase 1

#include "backends/rvv/include/sparse_attention_rvv.h"
#include <stdint.h>

/**
 * Generated sparse attention function
 * 
 * @param Q Query tensor [B x H x L x D]
 * @param K Key tensor [B x H x L x D]
 * @param V Value tensor [B x H x L x D]
 * @param O Output tensor [B x H x L x D]
 * @param B Batch size (default: {B})
 * @param H Number of heads (default: {H})
 * @param L Sequence length (default: {L})
 * @param D Head dimension (default: {D})
 */
void sparse_attention_generated(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int L,
    int D
) {{
    // Shape configuration
    sattn_shape_t shape = {{
        .batch = B,
        .heads = H,
        .seq_len = L,
        .head_dim = D
    }};
    
    // Pattern-specific parameters
    {params_struct} params = {params_init};
    
    // Call the validated RVV backend
    {func_name}(Q, K, V, O, shape, params{scales});
}}

// Convenience wrapper with default sizes
void sparse_attention_default(
    const float* Q,
    const float* K,
    const float* V,
    float* O
) {{
    sparse_attention_generated(Q, K, V, O, {B}, {H}, {L}, {D});
}}

/**
 * Usage example:
 * 
 * #include <stdlib.h>
 * 
 * int main() {{
 *     // Allocate tensors
 *     float* Q = (float*)malloc({B}*{H}*{L}*{D} * sizeof(float));
 *     float* K = (float*)malloc({B}*{H}*{L}*{D} * sizeof(float));
 *     float* V = (float*)malloc({B}*{H}*{L}*{D} * sizeof(float));
 *     float* O = (float*)malloc({B}*{H}*{L}*{D} * sizeof(float));
 *     
 *     // Initialize Q, K, V with your data...
 *     
 *     // Run sparse attention
 *     sparse_attention_default(Q, K, V, O);
 *     
 *     // Use output O...
 *     
 *     // Cleanup
 *     free(Q); free(K); free(V); free(O);
 *     return 0;
 * }}
 */
"""
    
    return c_code


def get_rvv_function_name(pattern: str, precision: str) -> str:
    """Get RVV C function name"""
    base = "sattn_rvv_"
    
    if pattern == "sliding_window":
        base += "sliding_global"
    elif pattern == "block_local_global":
        base += "blocktopk"
    elif pattern == "nm_structured":
        base += "nm_structured"
    elif pattern == "lsh":
        base += "lsh"
    elif pattern == "landmark":
        base += "landmark"
    
    if precision != "fp32":
        base += f"_{precision}"
    
    return base


def get_params_struct_name(pattern: str) -> str:
    """Get parameter struct type name"""
    if pattern == "sliding_window":
        return "sattn_params_t"
    elif pattern == "block_local_global":
        return "sattn_blocktopk_params_t"
    elif pattern == "nm_structured":
        return "sattn_nm_structured_params_t"
    elif pattern == "lsh":
        return "sattn_lsh_params_t"
    elif pattern == "landmark":
        return "sattn_landmark_params_t"
    return "sattn_params_t"


def get_params_initialization(pattern: str, **kwargs) -> str:
    """Generate parameter struct initialization"""
    if pattern == "sliding_window":
        window = kwargs.get("window_size", 16)
        return f"{{\n        .window_size = {window}\n    }}"
    elif pattern == "block_local_global":
        block = kwargs.get("block_size", 16)
        keep = kwargs.get("keep_ratio", 0.10)
        global_tok = kwargs.get("global_tokens", 4)
        return f"""{{\n        .block_size = {block},\n        .keep_ratio = {keep}f,\n        .global_tokens = {global_tok},\n        .gqa_group_size = 1,\n        .comp_block_size = {block}\n    }}"""
    elif pattern == "nm_structured":
        n = kwargs.get("nm_n", 2)
        m = kwargs.get("nm_m", 4)
        return f"{{\n        .n = {n},\n        .m = {m}\n    }}"
    elif pattern == "lsh":
        buckets = kwargs.get("buckets", 8)
        return f"{{\n        .buckets = {buckets}\n    }}"
    elif pattern == "landmark":
        landmarks = kwargs.get("num_landmarks", 16)
        return f"{{\n        .num_landmarks = {landmarks}\n    }}"
    return "{}"


# ===================================================================
# Main Compiler Interface
# ===================================================================

def compile_sparse_attention(
    pattern: str,
    precision: str,
    output_file: Optional[str] = None,
    **kwargs
) -> str:
    """
    Complete compilation pipeline.
    
    Args:
        pattern: Sparse attention pattern
        precision: Precision level (fp32/bf16/i8/i4)
        output_file: Optional output C file path
        **kwargs: Pattern-specific parameters
    
    Returns:
        Generated C code as string
    """
    
    print(f"🚀 Compiling {pattern} with {precision} precision...")
    
    # Step 1: Generate MLIR
    print("  [1/3] Generating MLIR...")
    mlir_code = generate_mlir_from_pytorch(pattern, precision, **kwargs)
    
    # Step 2: Apply lowering passes (simulated)
    print("  [2/3] Applying lowering passes...")
    # In a full implementation, this would call sattn-opt with passes
    # For now, we skip directly to C generation
    
    # Step 3: Generate C code
    print("  [3/3] Generating C code...")
    c_code = generate_c_code(pattern, precision, **kwargs)
    
    # Write to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(c_code)
        print(f"✅ Generated {output_file}")
    
    return c_code


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compile PyTorch sparse attention to RISC-V C code (Phase 2)"
    )
    
    parser.add_argument("--pattern", required=True,
                       choices=["sliding_window", "block_local_global", 
                               "nm_structured", "lsh", "landmark"],
                       help="Sparse attention pattern")
    parser.add_argument("--precision", required=True,
                       choices=["fp32", "bf16", "i8", "i4"],
                       help="Precision level")
    parser.add_argument("--output", "-o", required=True,
                       help="Output C file")
    
    # Shape parameters
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--L", type=int, default=128, help="Sequence length")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    
    # Pattern-specific parameters
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--global-tokens", type=int, default=4)
    parser.add_argument("--nm-n", type=int, default=2)
    parser.add_argument("--nm-m", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--num-landmarks", type=int, default=16)
    
    # Quantization
    parser.add_argument("--scale-q", type=float)
    parser.add_argument("--scale-k", type=float)
    parser.add_argument("--scale-v", type=float)
    
    args = parser.parse_args()
    
    # Collect kwargs
    kwargs = {
        "B": args.B, "H": args.H, "L": args.L, "D": args.D,
        "window_size": args.window_size,
        "block_size": args.block_size,
        "keep_ratio": args.keep_ratio,
        "global_tokens": args.global_tokens,
        "nm_n": args.nm_n,
        "nm_m": args.nm_m,
        "buckets": args.buckets,
        "num_landmarks": args.num_landmarks,
    }
    
    if args.scale_q:
        kwargs["scale_q"] = args.scale_q
    if args.scale_k:
        kwargs["scale_k"] = args.scale_k
    if args.scale_v:
        kwargs["scale_v"] = args.scale_v
    
    # Compile
    compile_sparse_attention(
        args.pattern,
        args.precision,
        args.output,
        **kwargs
    )
    
    print("\n📝 Next steps:")
    print(f"  1. Compile: gcc {args.output} -o attention -Ibackends/rvv/include -Lbuild/rvv-riscv64 -lsparse_attention_rvv")
    print(f"  2. Run: qemu-riscv64 -L /usr/riscv64-linux-gnu ./attention")


if __name__ == "__main__":
    main()

