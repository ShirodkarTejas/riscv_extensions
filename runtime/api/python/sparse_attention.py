from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from ops.sparse_attention.cpu import sparse_attention_cpu
try:
    from ops.sparse_attention.gpu import (
        sparse_attention_triton,
        sparse_attention_block_topk_torch,
        sparse_attention_triton_block_topk,
    )
    _HAS_TRITON_PATH = True
except Exception:
    _HAS_TRITON_PATH = False
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

Pattern = Literal["block_topk", "sliding_global"]
Precision = Literal["bf16", "fp16", "int8"]


@dataclass(frozen=True)
class SparseAttentionParams:
    block_size: Optional[int] = None  # for block_topk
    keep_ratio: Optional[float] = None  # for block_topk
    window_size: Optional[int] = None  # for sliding_global
    global_tokens: int = 0


def sparse_attention(
    Q: Any,
    K: Any,
    V: Any,
    pattern: Pattern,
    params: Dict[str, Any],
    precision: Precision = "bf16",
    training: bool = False,
) -> Any:
    """SparseAttention(Q, K, V, ...) -> O

    Contract (high-level):
      - Q, K, V shaped [B, H, L, D] with heads-contiguous layout.
      - pattern: "block_topk" or "sliding_global".
      - params: keys depend on pattern (see docs).
      - precision: bf16/fp16 (int8 later; dense fallback path exists).
      - training: if True, enables ops compatible with backward.
    """
    # GPU fast paths
    if _HAS_TORCH and _HAS_TRITON_PATH:
        is_torch = isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
        if is_torch and Q.is_cuda and K.is_cuda and V.is_cuda:
            if pattern == "sliding_global":
                return sparse_attention_triton(Q, K, V, pattern, params, precision, training)
            if pattern == "block_topk":
                # Prefer Triton gather kernel; fallback to Torch path if it fails
                try:
                    return sparse_attention_triton_block_topk(Q, K, V, params, precision, training)
                except Exception:
                    return sparse_attention_block_topk_torch(Q, K, V, params, precision, training)
    # CPU reference
    return sparse_attention_cpu(Q, K, V, pattern, params, precision, training)


