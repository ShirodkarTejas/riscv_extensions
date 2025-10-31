from typing import Any, Dict, List

import math


def sparse_attention_block_topk_torch(
    Q: Any,
    K: Any,
    V: Any,
    params: Dict[str, Any],
    precision: str = "bf16",
    training: bool = False,
):
    import torch

    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    B, H, L, D = Q.shape
    block_size = int(params.get("block_size", 64))
    keep_ratio = float(params.get("keep_ratio", 0.12))
    global_tokens = int(params.get("global_tokens", 0))

    num_blocks = (L + block_size - 1) // block_size
    k_blocks = max(1, math.ceil(keep_ratio * num_blocks))

    # Compute K block means: [B, H, num_blocks, D]
    pad = num_blocks * block_size - L
    if pad > 0:
        pad_k = torch.nn.functional.pad(K, (0, 0, 0, pad))  # pad on L dimension
    else:
        pad_k = K
    K_blocks = pad_k.view(B, H, num_blocks, block_size, D).mean(dim=3)

    # Compute block scores per row: [B, H, L, num_blocks]
    # scores[b,h,l,nb] = dot(Q[b,h,l,:], K_blocks[b,h,nb,:])
    scores = torch.einsum("bhld,bhnd->bhln", Q, K_blocks)

    # determine selected blocks per (b,h,l)
    topk_vals, topk_idx = torch.topk(scores, k=k_blocks, dim=-1, largest=True, sorted=False)  # [B,H,L,k]

    # add global tokens as blocks
    if global_tokens > 0:
        global_block_ids = (torch.arange(min(global_tokens, L), device=Q.device) // block_size).unique()
    else:
        global_block_ids = torch.tensor([], device=Q.device, dtype=topk_idx.dtype)

    # Prepare output
    O = torch.zeros_like(Q, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    # map token -> block id
    token_block_id = (torch.arange(L, device=Q.device) // block_size).to(topk_idx.dtype)  # [L]

    # Iterate per (b,h) and l for clarity; baseline correctness-focused
    for b in range(B):
        for h in range(H):
            Kh = K[b, h].to(torch.float32)
            Vh = V[b, h].to(torch.float32)
            for l in range(L):
                sel_blocks = topk_idx[b, h, l]  # [k]
                if global_block_ids.numel() > 0:
                    sel_blocks = torch.unique(torch.cat([sel_blocks, global_block_ids]))
                # build token mask by block membership
                mask = (token_block_id[:, None] == sel_blocks[None, :]).any(dim=1)  # [L]
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                q = Q[b, h, l, :].to(torch.float32)  # [D]
                subK = Kh.index_select(0, idx)  # [S,D]
                subV = Vh.index_select(0, idx)  # [S,D]
                att = (subK @ q) * scale  # [S]
                m = torch.max(att)
                p = torch.exp(att - m)
                p = p / (torch.sum(p) + 1e-12)
                O[b, h, l, :] = p @ subV

    if precision == "fp16":
        return O.to(torch.float16)
    if precision == "bf16":
        return O.to(torch.bfloat16)
    return O


