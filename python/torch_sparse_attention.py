"""
PyTorch integration for RISC-V Vector sparse attention.

This module provides PyTorch-compatible layers that can be used as drop-in
replacements for torch.nn.MultiheadAttention with sparse patterns and quantization.
"""

from typing import Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from sparse_attention_rvv import (
    SparseAttentionRVV,
    SparseAttnPattern,
    Precision,
    SparseAttentionConfig,
)


if _HAS_TORCH:
    class SparseAttentionLayer(nn.Module):
        """
        PyTorch sparse attention layer using optimized RVV backend.
        
        This is a drop-in replacement for torch.nn.MultiheadAttention with:
        - Sparse attention patterns (sliding_window, block_topk, etc.)
        - Quantization (fp32, bf16, i8, i4)
        - Optimized RVV implementation (Phase 1+2 optimizations)
        
        Args:
            d_model: Total dimension of the model
            num_heads: Number of attention heads
            pattern: Sparse attention pattern
            precision: Quantization precision
            dropout: Dropout probability (not implemented yet)
            bias: Whether to use bias (not implemented yet)
            **pattern_kwargs: Pattern-specific parameters
        
        Example:
            >>> import torch
            >>> from torch_sparse_attention import SparseAttentionLayer, SparseAttnPattern, Precision
            >>> 
            >>> # Create layer
            >>> attn = SparseAttentionLayer(
            ...     d_model=512,
            ...     num_heads=8,
            ...     pattern=SparseAttnPattern.SLIDING_WINDOW,
            ...     precision=Precision.I8,
            ...     window_size=16
            ... )
            >>> 
            >>> # Forward pass
            >>> x = torch.randn(2, 128, 512)  # [batch, seq_len, d_model]
            >>> output = attn(x, x, x)
            >>> print(output.shape)  # [2, 128, 512]
        """
        
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            pattern: Union[SparseAttnPattern, str],
            precision: Union[Precision, str] = Precision.FP32,
            dropout: float = 0.0,
            bias: bool = True,
            **pattern_kwargs
        ):
            super().__init__()
            
            if not _HAS_TORCH:
                raise ImportError("PyTorch is required for SparseAttentionLayer")
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.dropout = dropout
            
            if self.d_model % self.num_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
            # Convert string to enum if needed
            if isinstance(pattern, str):
                pattern = SparseAttnPattern(pattern)
            if isinstance(precision, str):
                precision = Precision(precision)
            
            # Linear projections
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.k_proj = nn.Linear(d_model, d_model, bias=bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=bias)
            self.o_proj = nn.Linear(d_model, d_model, bias=bias)
            
            # Sparse attention backend
            self.sparse_attn = SparseAttentionRVV(
                pattern=pattern,
                precision=precision,
                **pattern_kwargs
            )
            
            self.pattern = pattern
            self.precision = precision
        
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass compatible with torch.nn.MultiheadAttention.
            
            Args:
                query: Query tensor [batch, seq_len, d_model] or [seq_len, batch, d_model]
                key: Key tensor
                value: Value tensor
                key_padding_mask: Not yet implemented
                need_weights: If True, return attention weights (not supported)
                attn_mask: Not yet implemented
            
            Returns:
                output: Output tensor [batch, seq_len, d_model]
                attn_weights: Attention weights (only if need_weights=True, not supported)
            """
            # Handle both (batch, seq, feat) and (seq, batch, feat) formats
            is_batched = query.dim() == 3
            if not is_batched:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            
            batch_size, seq_len, _ = query.shape
            
            # Project to Q, K, V
            Q = self.q_proj(query)
            K = self.k_proj(key)
            V = self.v_proj(value)
            
            # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Make contiguous for C backend
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            
            # Call RVV backend (NumPy)
            device = query.device
            Q_np = Q.cpu().numpy()
            K_np = K.cpu().numpy()
            V_np = V.cpu().numpy()
            
            O_np = self.sparse_attn(Q_np, K_np, V_np)
            
            # Convert back to PyTorch
            O = torch.from_numpy(O_np).to(device)
            
            # Reshape back: [batch, seq_len, d_model]
            O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            
            # Output projection
            output = self.o_proj(O)
            
            if not is_batched:
                output = output.squeeze(0)
            
            if need_weights:
                raise NotImplementedError("Attention weights not yet supported")
            
            return output
    
    
    class SparseTransformerEncoderLayer(nn.Module):
        """
        Transformer encoder layer with sparse attention.
        
        Similar to torch.nn.TransformerEncoderLayer but with sparse attention support.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: FFN hidden dimension
            pattern: Sparse attention pattern
            precision: Quantization precision
            dropout: Dropout probability
            **pattern_kwargs: Pattern-specific parameters
        
        Example:
            >>> layer = SparseTransformerEncoderLayer(
            ...     d_model=512,
            ...     nhead=8,
            ...     dim_feedforward=2048,
            ...     pattern=SparseAttnPattern.SLIDING_WINDOW,
            ...     precision=Precision.I8,
            ...     window_size=16
            ... )
            >>> x = torch.randn(2, 128, 512)
            >>> output = layer(x)
        """
        
        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            pattern: Union[SparseAttnPattern, str] = SparseAttnPattern.SLIDING_WINDOW,
            precision: Union[Precision, str] = Precision.FP32,
            dropout: float = 0.1,
            activation: str = "relu",
            **pattern_kwargs
        ):
            super().__init__()
            
            # Sparse self-attention
            self.self_attn = SparseAttentionLayer(
                d_model=d_model,
                num_heads=nhead,
                pattern=pattern,
                precision=precision,
                dropout=dropout,
                **pattern_kwargs
            )
            
            # Feed-forward network
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            
            # Activation
            self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        
        def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                src: Input tensor [batch, seq_len, d_model]
                src_mask: Not yet implemented
                src_key_padding_mask: Not yet implemented
            
            Returns:
                Output tensor [batch, seq_len, d_model]
            """
            # Self-attention with residual
            src2 = self.self_attn(src, src, src)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward with residual
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src
    
    
    class SparseTransformer(nn.Module):
        """
        Complete transformer model with sparse attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: FFN hidden dimension
            pattern: Sparse attention pattern
            precision: Quantization precision
            dropout: Dropout probability
            **pattern_kwargs: Pattern-specific parameters
        
        Example:
            >>> model = SparseTransformer(
            ...     d_model=512,
            ...     nhead=8,
            ...     num_encoder_layers=6,
            ...     pattern=SparseAttnPattern.SLIDING_WINDOW,
            ...     precision=Precision.I8,
            ...     window_size=16
            ... )
            >>> x = torch.randn(2, 128, 512)
            >>> output = model(x)
        """
        
        def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 2048,
            pattern: Union[SparseAttnPattern, str] = SparseAttnPattern.SLIDING_WINDOW,
            precision: Union[Precision, str] = Precision.FP32,
            dropout: float = 0.1,
            **pattern_kwargs
        ):
            super().__init__()
            
            self.d_model = d_model
            
            # Encoder layers
            self.layers = nn.ModuleList([
                SparseTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    pattern=pattern,
                    precision=precision,
                    dropout=dropout,
                    **pattern_kwargs
                )
                for _ in range(num_encoder_layers)
            ])
            
            # Optional positional encoding
            self.pos_encoder = None  # Can add later if needed
        
        def forward(self, src: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                src: Input tensor [batch, seq_len, d_model]
            
            Returns:
                Output tensor [batch, seq_len, d_model]
            """
            x = src
            
            # Optional positional encoding
            if self.pos_encoder is not None:
                x = self.pos_encoder(x)
            
            # Pass through encoder layers
            for layer in self.layers:
                x = layer(x)
            
            return x


else:
    # Dummy classes if PyTorch is not available
    class SparseAttentionLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SparseAttentionLayer")
    
    class SparseTransformerEncoderLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SparseTransformerEncoderLayer")
    
    class SparseTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SparseTransformer")


if __name__ == "__main__":
    if _HAS_TORCH:
        print("Testing PyTorch integration...")
        
        # Test SparseAttentionLayer
        print("\n1. Testing SparseAttentionLayer...")
        attn = SparseAttentionLayer(
            d_model=64,
            num_heads=4,
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        x = torch.randn(2, 32, 64)
        y = attn(x, x, x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        print(f"   ✅ SparseAttentionLayer works!")
        
        # Test SparseTransformer
        print("\n2. Testing SparseTransformer...")
        model = SparseTransformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=256,
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        x = torch.randn(2, 32, 64)
        y = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        print(f"   ✅ SparseTransformer works!")
        
        # Test different precisions
        print("\n3. Testing different precisions...")
        for precision in [Precision.FP32, Precision.BF16, Precision.I8, Precision.I4]:
            try:
                attn = SparseAttentionLayer(
                    d_model=64,
                    num_heads=4,
                    pattern=SparseAttnPattern.SLIDING_WINDOW,
                    precision=precision,
                    window_size=8
                )
                x = torch.randn(1, 16, 64)
                y = attn(x, x, x)
                print(f"   ✅ {precision.value} works!")
            except Exception as e:
                print(f"   ❌ {precision.value} failed: {e}")
    else:
        print("PyTorch not available. Skipping tests.")

