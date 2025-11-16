// Example MLIR demonstrating all 5 patterns × 4 precisions
// This shows the enhanced sattn dialect supporting all configurations

module {
  //===--------------------------------------------------------------------===//
  // Pattern 1: Sliding Window (Most Energy Efficient)
  //===--------------------------------------------------------------------===//
  
  // Ultra-low-power: i4 precision (84% energy savings!)
  func.func @sliding_window_i4(%q: tensor<1x8x128x64xf32>,
                                %k: tensor<1x8x128x64xf32>,
                                %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.sliding_window(%q, %k, %v) {
      window_size = 16 : i64,
      precision = "i4",
      scale_q = 0.008 : f32,
      scale_k = 0.008 : f32,
      scale_v = 0.008 : f32
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }
  
  // Low-power: i8 precision (81% energy savings)
  func.func @sliding_window_i8(%q: tensor<1x8x128x64xf32>,
                                %k: tensor<1x8x128x64xf32>,
                                %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.sliding_window(%q, %k, %v) {
      window_size = 16 : i64,
      precision = "i8",
      scale_q = 0.020 : f32,
      scale_k = 0.020 : f32,
      scale_v = 0.020 : f32
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }
  
  // Balanced: bf16 precision (51% energy savings, fastest!)
  func.func @sliding_window_bf16(%q: tensor<1x8x128x64xf32>,
                                  %k: tensor<1x8x128x64xf32>,
                                  %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.sliding_window(%q, %k, %v) {
      window_size = 16 : i64,
      precision = "bf16"
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }
  
  // High-performance: fp32 precision (full accuracy)
  func.func @sliding_window_fp32(%q: tensor<1x8x128x64xf32>,
                                  %k: tensor<1x8x128x64xf32>,
                                  %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.sliding_window(%q, %k, %v) {
      window_size = 32 : i64,
      precision = "fp32"
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Pattern 2: Block-Local-Global (Best Efficiency: 12.0M GOPs/W)
  //===--------------------------------------------------------------------===//
  
  func.func @block_local_global_i4(%q: tensor<1x8x128x64xf32>,
                                    %k: tensor<1x8x128x64xf32>,
                                    %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.block_local_global(%q, %k, %v) {
      block_size = 16 : i64,
      keep_ratio = 0.08 : f32,
      global_tokens = 4 : i64,
      precision = "i4",
      scale_q = 0.008 : f32,
      scale_k = 0.008 : f32,
      scale_v = 0.008 : f32
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Pattern 3: N:M Structured Sparsity
  //===--------------------------------------------------------------------===//
  
  func.func @nm_structured_i8(%q: tensor<1x8x128x64xf32>,
                               %k: tensor<1x8x128x64xf32>,
                               %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.nm_structured(%q, %k, %v) {
      nm_n = 2 : i64,
      nm_m = 4 : i64,
      precision = "i8",
      scale_q = 0.020 : f32,
      scale_k = 0.020 : f32,
      scale_v = 0.020 : f32
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Pattern 4: LSH (Locality-Sensitive Hashing)
  //===--------------------------------------------------------------------===//
  
  func.func @lsh_bf16(%q: tensor<1x8x128x64xf32>,
                       %k: tensor<1x8x128x64xf32>,
                       %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.lsh(%q, %k, %v) {
      buckets = 8 : i64,
      precision = "bf16"
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Pattern 5: Landmark (Fastest Cycles)
  //===--------------------------------------------------------------------===//
  
  func.func @landmark_fp32(%q: tensor<1x8x128x64xf32>,
                            %k: tensor<1x8x128x64xf32>,
                            %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    %output = sattn.landmark(%q, %k, %v) {
      num_landmarks = 16 : i64,
      precision = "fp32"
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Generic Op (Pattern Selection at Compile Time)
  //===--------------------------------------------------------------------===//
  
  func.func @generic_sparse_attention(%q: tensor<1x8x128x64xf32>,
                                       %k: tensor<1x8x128x64xf32>,
                                       %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    // Generic op - will be lowered to specific pattern based on cost model
    %output = sattn.sparse_attention(%q, %k, %v) {
      pattern = "sliding_window",
      window_size = 16 : i64,
      precision = "i8",
      scale_q = 0.020 : f32,
      scale_k = 0.020 : f32,
      scale_v = 0.020 : f32
    } : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      -> tensor<1x8x128x64xf32>
    return %output : tensor<1x8x128x64xf32>
  }

  //===--------------------------------------------------------------------===//
  // Example: After Lowering to RVV
  //===--------------------------------------------------------------------===//
  
  func.func @lowered_to_rvv(%q: memref<1x8x128x64xf32>,
                             %k: memref<1x8x128x64xf32>,
                             %v: memref<1x8x128x64xf32>,
                             %o: memref<1x8x128x64xf32>) {
    // After bufferization and lowering, this becomes a call to our C function
    sattn.rvv_call "sattn_rvv_sliding_global_i8"(%q, %k, %v, %o) {
      pattern = "sliding_window",
      precision = "i8",
      window_size = 16 : i64,
      scale_q = 0.020 : f32,
      scale_k = 0.020 : f32,
      scale_v = 0.020 : f32
    } : (memref<1x8x128x64xf32>, memref<1x8x128x64xf32>, 
         memref<1x8x128x64xf32>, memref<1x8x128x64xf32>) -> ()
    return
  }
}

