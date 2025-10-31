from typing import Any, Dict

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _ensure_triton():
    if not _HAS_TRITON:
        raise ImportError("Triton is not available. Install triton to use GPU kernels.")


def _compute_block_topk_indices(Q, K, block_size: int, keep_ratio: float, global_tokens: int):
    import torch
    B, H, L, D = Q.shape
    num_blocks = (L + block_size - 1) // block_size
    k_blocks = max(1, int((num_blocks * keep_ratio + 0.9999)))
    pad = num_blocks * block_size - L
    if pad > 0:
        Kp = torch.nn.functional.pad(K, (0, 0, 0, pad))
    else:
        Kp = K
    K_blocks = Kp.view(B, H, num_blocks, block_size, D).mean(dim=3)  # [B,H,num_blocks,D]
    # scores[b,h,l,nb] = dot(Q[b,h,l,:], K_blocks[b,h,nb,:])
    scores = torch.einsum("bhld,bhnd->bhln", Q, K_blocks)  # [B,H,L,num_blocks]
    _, topk_idx = torch.topk(scores, k=k_blocks, dim=-1, largest=True, sorted=False)  # [B,H,L,k]
    # expand block ids to token ids
    token_ids = torch.arange(num_blocks * block_size, device=Q.device).view(num_blocks, block_size)  # [NB,BS]
    sel = token_ids[topk_idx]  # [B,H,L,k,block_size]
    sel = sel.reshape(B, H, L, -1)  # [B,H,L,S]
    if global_tokens > 0:
        gb = torch.arange(min(global_tokens, L), device=Q.device)
        sel = torch.cat([sel, gb.view(1, 1, 1, -1).expand(B, H, L, -1)], dim=-1)
    # clamp to L
    sel = torch.clamp(sel, max=L - 1)
    return sel  # int64 [B,H,L,S]


@triton.jit
def _gattn_fwd(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, O_ptr,
    B, H, L, D, S,
    stride_q_b, stride_q_h, stride_q_l, stride_q_d,
    stride_k_b, stride_k_h, stride_k_l, stride_k_d,
    stride_v_b, stride_v_h, stride_v_l, stride_v_d,
    stride_o_b, stride_o_h, stride_o_l, stride_o_d,
    stride_idx_b, stride_idx_h, stride_idx_l, stride_idx_s,
    BLOCK_M: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile [M,D]
    q_ptrs = Q_ptr + (
        b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_l + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D), other=0.0)

    scale = 1.0 / tl.sqrt(tl.float32(D))
    # streaming softmax stats
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # First pass: compute m_i and l_i across S in chunks
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        # idx [M,BS]
        idx_ptrs = IDX_ptr + (
            b * stride_idx_b + h * stride_idx_h + offs_m[:, None] * stride_idx_l + offs_s[None, :] * stride_idx_s
        )
        idx = tl.load(idx_ptrs, mask=(offs_m[:, None] < L) & (offs_s[None, :] < S), other=0)
        # gather K: pointers shape [M,BS,D]
        k_ptrs = K_ptr + (
            b * stride_k_b + h * stride_k_h + idx[:, :, None] * stride_k_l + offs_d[None, None, :] * stride_k_d
        )
        k = tl.load(k_ptrs, mask=(offs_m[:, None, None] < L) & (offs_s[None, :, None] < S) & (offs_d[None, None, :] < D), other=0.0)
        # att[m,bs] = sum_d(q[m,d]*k[m,bs,d]) * scale
        att = tl.sum(k * q[:, None, :], axis=2) * scale
        m_ij = tl.max(att, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(att - m_new[:, None]), axis=1)
        m_i = m_new

    # Second pass: compute probabilities and accumulate O
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        idx_ptrs = IDX_ptr + (
            b * stride_idx_b + h * stride_idx_h + offs_m[:, None] * stride_idx_l + offs_s[None, :] * stride_idx_s
        )
        idx = tl.load(idx_ptrs, mask=(offs_m[:, None] < L) & (offs_s[None, :] < S), other=0)
        k_ptrs = K_ptr + (
            b * stride_k_b + h * stride_k_h + idx[:, :, None] * stride_k_l + offs_d[None, None, :] * stride_k_d
        )
        v_ptrs = V_ptr + (
            b * stride_v_b + h * stride_v_h + idx[:, :, None] * stride_v_l + offs_d[None, None, :] * stride_v_d
        )
        k = tl.load(k_ptrs, mask=(offs_m[:, None, None] < L) & (offs_s[None, :, None] < S) & (offs_d[None, None, :] < D), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_m[:, None, None] < L) & (offs_s[None, :, None] < S) & (offs_d[None, None, :] < D), other=0.0)
        att = tl.sum(k * q[:, None, :], axis=2) * scale
        p = tl.exp(att - m_i[:, None]) / l_i[:, None]
        # acc += p @ v  -> [M,D]
        acc += tl.dot(p, v)

    o_ptrs = O_ptr + (
        b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_l + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D))


def sparse_attention_triton_block_topk(
    Q: Any,
    K: Any,
    V: Any,
    params: Dict[str, Any],
    precision: str = "bf16",
    training: bool = False,
):
    _ensure_triton()
    import torch
    Q_t = torch.as_tensor(Q).contiguous().cuda()
    K_t = torch.as_tensor(K).contiguous().cuda()
    V_t = torch.as_tensor(V).contiguous().cuda()
    B, H, L, D = Q_t.shape
    block_size = int(params.get("block_size", 64))
    keep_ratio = float(params.get("keep_ratio", 0.12))
    global_tokens = int(params.get("global_tokens", 0))
    idx = _compute_block_topk_indices(Q_t, K_t, block_size, keep_ratio, global_tokens).contiguous()  # [B,H,L,S]
    S = idx.size(-1)

    O_t = torch.empty_like(Q_t, dtype=torch.float32)
    grid = ((L + 63) // 64, B * H)
    _gattn_fwd[grid](
        Q_t, K_t, V_t, idx, O_t,
        B, H, L, D, S,
        Q_t.stride(0), Q_t.stride(1), Q_t.stride(2), Q_t.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
        V_t.stride(0), V_t.stride(1), V_t.stride(2), V_t.stride(3),
        O_t.stride(0), O_t.stride(1), O_t.stride(2), O_t.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        BLOCK_M=64, BLOCK_S=min(S, 128), BLOCK_D=min(D, 128),
        num_warps=4, num_stages=2,
    )

    if precision == "fp16":
        return O_t.to(torch.float16)
    if precision == "bf16":
        return O_t.to(torch.bfloat16)
    return O_t


