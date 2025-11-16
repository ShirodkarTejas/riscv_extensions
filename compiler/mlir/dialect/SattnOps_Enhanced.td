//===- SattnOps_Enhanced.td - SATTN Dialect Enhanced Ops ------*- tablegen -*-===//
// Enhanced MLIR TableGen spec for sparse attention supporting:
// - All 5 patterns (sliding_window, block_local_global, nm_structured, lsh, landmark)
// - All 4 precisions (fp32, bf16, i8, i4)
// - Quantization parameters
// - Pattern-specific attributes
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Sattn_Dialect : Dialect {
  let name = "sattn";
  let summary = "Sparse attention operators for efficient transformer inference";
  let description = [{
    The SATTN dialect provides operations for various sparse attention patterns
    optimized for RISC-V backends (RVV and custom accelerators).
    
    Supported patterns:
    - sliding_window: Local windowed attention
    - block_local_global: Block-sparse with global tokens
    - nm_structured: N:M structured sparsity  
    - lsh: Locality-sensitive hashing
    - landmark: Landmark compression
    
    Supported precisions:
    - fp32: Full precision
    - bf16: Bfloat16
    - i8: Int8 quantization (with Phase 1+2 optimizations)
    - i4: Int4 quantization (with Phase 1+2 optimizations)
  }];
  let cppNamespace = "::mlir::sattn";
}

//===----------------------------------------------------------------------===//
// Base Sparse Attention Operation
//===----------------------------------------------------------------------===//

def Sattn_SparseAttentionOp : Op<Sattn_Dialect, "sparse_attention", [Pure]> {
  let summary = "Generic sparse attention operator";
  let description = [{
    The `sattn.sparse_attention` op computes sparse attention for Q,K,V tensors
    with configurable sparsity patterns and precision levels.
    
    Example:
    ```mlir
    %output = sattn.sparse_attention(%q, %k, %v) {
      pattern = "sliding_window",
      window_size = 16 : i64,
      precision = "i8",
      scale_q = 0.008 : f32,
      scale_k = 0.008 : f32,
      scale_v = 0.008 : f32
    } : tensor<1x8x128x64xf32> -> tensor<1x8x128x64xf32>
    ```
  }];

  let arguments = (ins
    // Operands
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$q,
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$k,
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$v,
    
    // Common attributes
    StrAttr:$pattern,
    StrAttr:$precision,
    
    // Sliding window attributes
    OptionalAttr<I64Attr>:$window_size,
    OptionalAttr<I64Attr>:$dilation,
    
    // Block-local-global attributes
    OptionalAttr<I64Attr>:$block_size,
    OptionalAttr<F32Attr>:$keep_ratio,
    OptionalAttr<I64Attr>:$global_tokens,
    
    // N:M structured attributes  
    OptionalAttr<I64Attr>:$nm_n,
    OptionalAttr<I64Attr>:$nm_m,
    
    // LSH attributes
    OptionalAttr<I64Attr>:$buckets,
    
    // Landmark attributes
    OptionalAttr<I64Attr>:$num_landmarks,
    
    // Quantization parameters (for i8/i4)
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v,
    OptionalAttr<I32Attr>:$zero_point_q,
    OptionalAttr<I32Attr>:$zero_point_k,
    OptionalAttr<I32Attr>:$zero_point_v,
    
    // Tiling hints (for optimization)
    OptionalAttr<I64Attr>:$tile_M,
    OptionalAttr<I64Attr>:$tile_D,
    OptionalAttr<I64Attr>:$tile_S
  );

  let results = (outs AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$output);

  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pattern-Specific Operations (for optimized lowering)
//===----------------------------------------------------------------------===//

def Sattn_SlidingWindowOp : Op<Sattn_Dialect, "sliding_window", [Pure]> {
  let summary = "Sliding window sparse attention";
  let description = [{
    Specialized op for sliding window pattern with local attention.
    
    Most efficient pattern for energy (84% savings with i4).
  }];

  let arguments = (ins
    AnyRankedTensor:$q,
    AnyRankedTensor:$k,
    AnyRankedTensor:$v,
    I64Attr:$window_size,
    StrAttr:$precision,
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
}

def Sattn_BlockLocalGlobalOp : Op<Sattn_Dialect, "block_local_global", [Pure]> {
  let summary = "Block-sparse attention with global tokens";
  let description = [{
    Block-sparse pattern with explicit global token support.
    
    Best efficiency (12.0M GOPs/W with i4).
  }];

  let arguments = (ins
    AnyRankedTensor:$q,
    AnyRankedTensor:$k,
    AnyRankedTensor:$v,
    I64Attr:$block_size,
    F32Attr:$keep_ratio,
    I64Attr:$global_tokens,
    StrAttr:$precision,
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
}

def Sattn_NMStructuredOp : Op<Sattn_Dialect, "nm_structured", [Pure]> {
  let summary = "N:M structured sparsity attention";
  let description = [{
    N:M structured sparsity pattern (e.g., 2:4, 4:8).
  }];

  let arguments = (ins
    AnyRankedTensor:$q,
    AnyRankedTensor:$k,
    AnyRankedTensor:$v,
    I64Attr:$nm_n,
    I64Attr:$nm_m,
    StrAttr:$precision,
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
}

def Sattn_LSHOp : Op<Sattn_Dialect, "lsh", [Pure]> {
  let summary = "Locality-sensitive hashing attention";
  let description = [{
    LSH-based sparse attention with hash buckets.
  }];

  let arguments = (ins
    AnyRankedTensor:$q,
    AnyRankedTensor:$k,
    AnyRankedTensor:$v,
    I64Attr:$buckets,
    StrAttr:$precision,
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
}

def Sattn_LandmarkOp : Op<Sattn_Dialect, "landmark", [Pure]> {
  let summary = "Landmark compression attention";
  let description = [{
    Landmark-based attention with sequence compression.
    
    Fastest cycles for dense compression.
  }];

  let arguments = (ins
    AnyRankedTensor:$q,
    AnyRankedTensor:$k,
    AnyRankedTensor:$v,
    I64Attr:$num_landmarks,
    StrAttr:$precision,
    OptionalAttr<F32Attr>:$scale_q,
    OptionalAttr<F32Attr>:$scale_k,
    OptionalAttr<F32Attr>:$scale_v
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $q `,` $k `,` $v `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Backend Lowering Target Operations
//===----------------------------------------------------------------------===//

def Sattn_RVVCallOp : Op<Sattn_Dialect, "rvv_call", []> {
  let summary = "Lowered call to RVV backend";
  let description = [{
    Represents a call to the RVV C backend after lowering.
    Carries all necessary parameters for code generation.
  }];

  let arguments = (ins
    StrAttr:$function_name,
    StrAttr:$pattern,
    StrAttr:$precision,
    Variadic<AnyType>:$operands
  );

  let results = (outs);

  let assemblyFormat = [{
    $function_name `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Sattn_RoCCCallOp : Op<Sattn_Dialect, "rocc_call", []> {
  let summary = "Lowered call to RoCC accelerator";
  let description = [{
    Represents a call to the custom RoCC accelerator.
    Used for hardware acceleration path.
  }];

  let arguments = (ins
    StrAttr:$command,
    Variadic<AnyType>:$operands
  );

  let results = (outs);

  let assemblyFormat = [{
    $command `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def Sattn_QuantizeOp : Op<Sattn_Dialect, "quantize", [Pure]> {
  let summary = "Quantize tensor from fp32 to i8/i4";
  let description = [{
    Quantizes a floating-point tensor to integer representation.
    
    Supports symmetric and asymmetric quantization.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    StrAttr:$target_type,  // "i8" or "i4"
    F32Attr:$scale,
    OptionalAttr<I32Attr>:$zero_point
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];
}

def Sattn_DequantizeOp : Op<Sattn_Dialect, "dequantize", [Pure]> {
  let summary = "Dequantize tensor from i8/i4 to fp32";
  let description = [{
    Dequantizes an integer tensor back to floating-point.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F32Attr:$scale,
    OptionalAttr<I32Attr>:$zero_point
  );

  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SATTN_OPS_ENHANCED

