from typing import Any, Dict, List, Tuple
import math

import numpy as np


def _validate_inputs(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[int, int, int, int]:
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4, "Q,K,V must be [B,H,L,D]"
    B, H, L, D = Q.shape
    assert K.shape == (B, H, L, D), "K must match Q shape"
    assert V.shape == (B, H, L, D), "V must match Q shape"
    return B, H, L, D


def _blocks_for_length(L: int, block_size: int) -> int:
    return (L + block_size - 1) // block_size


def _block_token_range(block_idx: int, block_size: int, L: int) -> Tuple[int, int]:
    start = block_idx * block_size
    end = min(L, start + block_size)
    return start, end


def _global_blocks(global_tokens: int, block_size: int, L: int) -> List[int]:
    if global_tokens <= 0:
        return []
    token_ids = list(range(min(global_tokens, L)))
    blocks = {tid // block_size for tid in token_ids}
    return sorted(blocks)


def _select_blocks_block_topk(
    q_row: np.ndarray,  # [D]
    K_row_blocks: np.ndarray,  # [num_blocks, block_len, D] ragged via masking implied by L
    block_size: int,
    keep_ratio: float,
    global_tokens: int,
    L: int,
) -> List[int]:
    num_blocks = _blocks_for_length(L, block_size)
    k_blocks = max(1, int(math.ceil(keep_ratio * num_blocks)))
    # score per block: dot(q, mean(K_block))
    scores = []
    for b in range(num_blocks):
        s, e = _block_token_range(b, block_size, L)
        if s >= e:
            scores.append((-1e30, b))
            continue
        kb = K_row_blocks[b, : (e - s), :]
        kb_mean = kb.mean(axis=0)
        scores.append((float(np.dot(q_row, kb_mean)), b))
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [b for _, b in scores[:k_blocks]]
    # union with global blocks
    gbs = _global_blocks(global_tokens, block_size, L)
    return sorted(set(selected).union(gbs))


def _select_blocks_sliding_global(
    i_token: int,
    window_size: int,
    global_tokens: int,
    block_size: int,
    L: int,
) -> List[int]:
    left = max(0, i_token - window_size)
    right = min(L, i_token + window_size + 1)
    blocks = set()
    for t in (left, right - 1):
        blocks.add(t // block_size)
    # include all blocks spanned by window
    start_block = left // block_size
    end_block = (right - 1) // block_size
    for b in range(start_block, end_block + 1):
        blocks.add(b)
    # union with global blocks
    for gb in _global_blocks(global_tokens, block_size, L):
        blocks.add(gb)
    return sorted(blocks)


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    y = np.exp(x - m)
    return y / (np.sum(y) + 1e-12)


def _row_attention(
    q_row: np.ndarray,  # [D]
    K_mat: np.ndarray,  # [L, D]
    V_mat: np.ndarray,  # [L, D]
    token_indices: List[int],
) -> np.ndarray:
    if len(token_indices) == 0:
        return np.zeros_like(q_row)
    subK = K_mat[token_indices, :]  # [S, D]
    subV = V_mat[token_indices, :]  # [S, D]
    scores = subK @ q_row  # [S]
    scores = scores / math.sqrt(q_row.shape[0])
    probs = _stable_softmax(scores)
    return probs @ subV  # [D]


def _materialize_K_blocks(K_mat: np.ndarray, block_size: int) -> np.ndarray:
    # Packs K blocks into shape [num_blocks, block_size, D], last block truncated by L
    L, D = K_mat.shape
    num_blocks = _blocks_for_length(L, block_size)
    K_blocks = np.zeros((num_blocks, block_size, D), dtype=K_mat.dtype)
    for b in range(num_blocks):
        s, e = _block_token_range(b, block_size, L)
        if s < e:
            K_blocks[b, : (e - s), :] = K_mat[s:e, :]
    return K_blocks


def _precision_cast(x: np.ndarray, precision: str) -> np.ndarray:
    if precision == "bf16":
        # numpy has no bf16 dtype universally; use float32 as proxy here
        return x.astype(np.float32)
    if precision == "fp16":
        return x.astype(np.float16)
    if precision == "int8":
        return x.astype(np.int8)
    return x


def sparse_attention_cpu(
    Q: Any,
    K: Any,
    V: Any,
    pattern: str,
    params: Dict[str, Any],
    precision: str = "bf16",
    training: bool = False,
) -> Any:
    Qn = np.asarray(Q)
    Kn = np.asarray(K)
    Vn = np.asarray(V)
    B, H, L, D = _validate_inputs(Qn, Kn, Vn)
    # cast for compute; accumulate in float32 regardless
    Qc = _precision_cast(Qn, precision).astype(np.float32)
    Kc = _precision_cast(Kn, precision).astype(np.float32)
    Vc = _precision_cast(Vn, precision).astype(np.float32)

    block_size = int(params.get("block_size", 64))
    keep_ratio = float(params.get("keep_ratio", 0.12))
    window_size = int(params.get("window_size", 512))
    global_tokens = int(params.get("global_tokens", 0))

    O = np.zeros_like(Qc)
    for b in range(B):
        for h in range(H):
            Qbh = Qc[b, h]  # [L, D]
            Kbh = Kc[b, h]
            Vbh = Vc[b, h]
            if pattern == "block_topk":
                K_blocks = _materialize_K_blocks(Kbh, block_size)
            else:
                K_blocks = None
            for i in range(L):
                q_row = Qbh[i]
                if pattern == "block_topk":
                    sel_blocks = _select_blocks_block_topk(
                        q_row, K_blocks, block_size, keep_ratio, global_tokens, L
                    )
                    token_indices: List[int] = []
                    for bidx in sel_blocks:
                        s, e = _block_token_range(bidx, block_size, L)
                        token_indices.extend(range(s, e))
                elif pattern == "sliding_global":
                    sel_blocks = _select_blocks_sliding_global(
                        i, window_size, global_tokens, block_size, L
                    )
                    token_indices = []
                    for bidx in sel_blocks:
                        s, e = _block_token_range(bidx, block_size, L)
                        token_indices.extend(range(s, e))
                else:
                    # dense fallback
                    token_indices = list(range(L))

                O[b, h, i, :] = _row_attention(q_row, Kbh, Vbh, token_indices)

    # cast back to requested precision for output (compute accuracy kept)
    if precision == "fp16":
        return O.astype(np.float16)
    # bf16 returned as float32 proxy
    return O.astype(np.float32)


# Adapters (placeholders for now)
def bsr_to_csr(bsr_indices: List[List[int]], block_size: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    # Return (indptr, indices) CSR-like for tokens by expanding blocks
    indptr = [0]
    indices: List[int] = []
    for i in range(L):
        # naive expansion: include full row; refined materialization left for future
        indices.extend(range(L))
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int64)


def coo_from_bsr(bsr_indices: List[List[int]], block_size: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[int] = []
    cols: List[int] = []
    for i in range(L):
        for bidx in bsr_indices[i]:
            s, e = _block_token_range(bidx, block_size, L)
            for t in range(s, e):
                rows.append(i)
                cols.append(t)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


