//===- SattnOps.td - SATTN Dialect Operation definitions -------*- tablegen -*-===//
// Minimal MLIR TableGen spec for sparse attention op. This is a documentation
// and scaffolding artifact; integration into an MLIR build is left to later stages.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def Sattn_Dialect : Dialect {
  let name = "sattn";
  let cppNamespace = "mlir::sattn";
}

def Sattn_SparseAttentionOp : Op<Sattn_Dialect, "sparse_attention", [NoSideEffect]> {
  let summary = "Sparse attention operator with block/top-k and sliding-global patterns";
  let description = [{
The `sattn.sparse_attention` op computes sparse attention for Q,K,V tensors.

Operands:
  - `q`, `k`, `v`: ranked tensors of shape [B, H, L, D]

Attributes:
  - `pattern`: string attr in {"block_topk", "sliding_global"}
  - `block_size`: i64 (tokens per block)
  - `keep_ratio`: f32 (0..1) fraction of blocks kept per row (block_topk)
  - `global_tokens`: i64 number of global tokens
  - `window_size`: i64 (sliding_global)
  - `precision`: string attr in {"bf16","fp16","int8"}
  - `softmax_mode`: string attr in {"logsumexp","none"}

Result:
  - output tensor of shape [B, H, L, D]
}];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$q,
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$k,
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$v,
    StrAttr:$pattern,
    I64Attr:$block_size,
    F32Attr:$keep_ratio,
    I64Attr:$global_tokens,
    I64Attr:$window_size,
    StrAttr:$precision,
    StrAttr:$softmax_mode
  );

  let results = (outs AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$o);

  let assemblyFormat = [{
    $q `,` $k `,` $v attr-dict `:` type($q) `,` type($k) `,` type($v) `->` type($o)
  }];
}

def Sattn_RoCCCallOp : Op<Sattn_Dialect, "rocc_call", [NoSideEffect]> {
  let summary = "Emit RoCC descriptor/call for sparse attention";
  let description = [{
The `sattn.rocc_call` op represents a lowered call to the RoCC accelerator.
It carries attributes describing the descriptor:
  - `m_rows`, `head_dim_d`, `s_tokens`, `block_size`
  - `indices` (sliding) or `block_indices` (block_topk)
  - `pattern`, `precision`
This is a placeholder lowering target used for tooling/prototyping.
}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{attr-dict}];
}

def Sattn_RVVCallOp : Op<Sattn_Dialect, "rvv_call", [NoSideEffect]> {
  let summary = "Emit RVV descriptor/call for sparse attention";
  let description = [{
The `sattn.rvv_call` op represents a lowered call/site for the RVV backend.
It carries attributes describing the chosen tile sizes and pattern settings:
  - `m_rows`, `head_dim_d`, `s_tokens`, `window_size`
  - `pattern`, `precision`
}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{attr-dict}];
}


