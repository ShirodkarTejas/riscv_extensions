from typing import Any, Dict

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


@triton.jit
def _swattn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H, L, D,
    stride_q_b, stride_q_h, stride_q_l, stride_q_d,
    stride_k_b, stride_k_h, stride_k_l, stride_k_d,
    stride_v_b, stride_v_h, stride_v_l, stride_v_d,
    stride_o_b, stride_o_h, stride_o_l, stride_o_d,
    window: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    # rows [m_start, m_start+BLOCK_M)
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # pointers to Q rows
    q_ptrs = Q_ptr + (
        b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_l + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D), other=0.0)

    scale = 1.0 / tl.sqrt(tl.float32(D))

    # streaming softmax stats
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # window bounds for each row
    # For each query row i, valid keys j in [i-window, i+window]
    # We'll iterate n blocks and mask out-of-window keys.

    for n_start in range(0, L, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        # load K tile [N,D]
        k_ptrs = K_ptr + (
            b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_l + offs_d[None, :] * stride_k_d
        )
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < L) & (offs_d[None, :] < D), other=0.0)
        # compute att = Q * K^T => [M,N]
        att = tl.dot(q, tl.trans(k)) * scale
        # mask by sliding window
        i_idx = offs_m[:, None]
        j_idx = offs_n[None, :]
        valid = (i_idx < L) & (j_idx < L) & (j_idx + 0 >= tl.maximum(0, i_idx - window)) & (j_idx <= tl.minimum(L - 1, i_idx + window))
        att = tl.where(valid, att, -float("inf"))
        # update m_i and l_i
        m_ij = tl.max(att, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # l_i = l_i*exp(m_i-m_new) + sum(exp(att-m_new))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(att - m_new[:, None]), axis=1)
        m_i = m_new

    # second pass: compute output
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n_start in range(0, L, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        # load K and V tiles
        k_ptrs = K_ptr + (
            b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_l + offs_d[None, :] * stride_k_d
        )
        v_ptrs = V_ptr + (
            b * stride_v_b + h * stride_v_h + offs_n[:, None] * stride_v_l + offs_d[None, :] * stride_v_d
        )
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < L) & (offs_d[None, :] < D), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < L) & (offs_d[None, :] < D), other=0.0)
        att = tl.dot(q, tl.trans(k)) * scale
        i_idx = offs_m[:, None]
        j_idx = offs_n[None, :]
        valid = (i_idx < L) & (j_idx < L) & (j_idx + 0 >= tl.maximum(0, i_idx - window)) & (j_idx <= tl.minimum(L - 1, i_idx + window))
        att = tl.where(valid, att, -float("inf"))
        p = tl.exp(att - m_i[:, None]) / l_i[:, None]
        acc += tl.dot(p, v)

    # store O
    o_ptrs = O_ptr + (
        b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_l + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D))


def sparse_attention_triton(
    Q: Any,
    K: Any,
    V: Any,
    pattern: str,
    params: Dict[str, Any],
    precision: str = "bf16",
    training: bool = False,
):
    if not _HAS_TRITON:
        raise ImportError("Triton is not available. Install triton to use GPU kernels.")
    import torch

    if pattern != "sliding_global":
        raise NotImplementedError("Triton path currently supports 'sliding_global' only.")

    window = int(params.get("window_size", 512))
    block_d = 64  # assume head_dim multiple of 64; adjust at call
    BLOCK_M = 64
    BLOCK_N = 128

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    Q_t = to_torch(Q).contiguous().cuda()
    K_t = to_torch(K).contiguous().cuda()
    V_t = to_torch(V).contiguous().cuda()

    assert Q_t.dtype in (torch.float16, torch.bfloat16, torch.float32)
    # compute in fp16/bf16; kernel promotes to fp32 internally
    B, H, L, D = Q_t.shape
    O_t = torch.empty_like(Q_t, dtype=torch.float32)

    grid = (
        (L + BLOCK_M - 1) // BLOCK_M,
        B * H,
    )

    _swattn_fwd[grid](
        Q_t, K_t, V_t, O_t,
        B, H, L, D,
        Q_t.stride(0), Q_t.stride(1), Q_t.stride(2), Q_t.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
        V_t.stride(0), V_t.stride(1), V_t.stride(2), V_t.stride(3),
        O_t.stride(0), O_t.stride(1), O_t.stride(2), O_t.stride(3),
        window=window,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=min(D, 128),
        num_warps=4,
        num_stages=2,
    )

    if precision == "fp16":
        return O_t.to(torch.float16)
    # bf16 as fp32 result by default; caller may cast
    return O_t


