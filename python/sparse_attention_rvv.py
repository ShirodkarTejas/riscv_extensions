"""
Python bindings for RISC-V Vector (RVV) sparse attention backend.

This module provides ctypes bindings to call the optimized RVV C implementations
directly from Python, enabling end-to-end testing and integration with PyTorch/JAX.
"""

import ctypes
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Try to find the shared library or static library
def _find_rvv_library():
    """Locate the RVV library (shared or static)"""
    # Try common build locations (both .so and .a)
    possible_paths = [
        "build/rvv-riscv64/libsparse_attention_rvv.so",
        "build/libsparse_attention_rvv.so",
        "/workspace/build/rvv-riscv64/libsparse_attention_rvv.so",
        "/workspace/backends/rvv/build/libsattn_rvv.a",  # Static lib in Docker
        "backends/rvv/build/libsattn_rvv.a",
        os.environ.get("SATTN_RVV_LIB", ""),
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            return str(Path(path).absolute())
    
    raise FileNotFoundError(
        "Could not find RVV library (libsparse_attention_rvv.so or libsattn_rvv.a). "
        "Please build the RVV backend first or set SATTN_RVV_LIB environment variable."
    )


# Load shared library
try:
    _lib = ctypes.CDLL(_find_rvv_library())
    _LIB_AVAILABLE = True
except (FileNotFoundError, OSError) as e:
    print(f"Warning: RVV library not available: {e}")
    _LIB_AVAILABLE = False
    _lib = None


# ============================================================================
# Enums
# ============================================================================

class SparseAttnPattern(Enum):
    """Sparse attention patterns"""
    SLIDING_WINDOW = "sliding_window"
    BLOCK_LOCAL_GLOBAL = "block_local_global"
    NM_STRUCTURED = "nm_structured"
    LSH = "lsh"
    LANDMARK = "landmark"


class Precision(Enum):
    """Precision levels for quantization"""
    FP32 = "fp32"
    BF16 = "bf16"
    I8 = "i8"
    I4 = "i4"


# ============================================================================
# C Structures
# ============================================================================

class SattnShapeT(ctypes.Structure):
    """C struct: sattn_shape_t"""
    _fields_ = [
        ("B", ctypes.c_int64),
        ("H", ctypes.c_int64),
        ("L", ctypes.c_int64),
        ("D", ctypes.c_int64),
    ]


class SattnParamsT(ctypes.Structure):
    """C struct: sattn_params_t (for sliding_window)"""
    _fields_ = [
        ("window_size", ctypes.c_int),
        ("block_size", ctypes.c_int),
        ("dilation", ctypes.c_int),
        ("wrap", ctypes.c_int),
    ]


class SattnBlockTopKParamsT(ctypes.Structure):
    """C struct: sattn_blocktopk_params_t"""
    _fields_ = [
        ("block_size", ctypes.c_int),
        ("keep_ratio", ctypes.c_float),
        ("global_tokens", ctypes.c_int),
        ("gqa_group_size", ctypes.c_int),
        ("comp_block_size", ctypes.c_int),
    ]


class SattnNMStructuredParamsT(ctypes.Structure):
    """C struct: sattn_nm_params_t"""
    _fields_ = [
        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
    ]


class SattnLSHParamsT(ctypes.Structure):
    """C struct: sattn_lsh_params_t"""
    _fields_ = [
        ("buckets", ctypes.c_int),
    ]


class SattnLandmarkParamsT(ctypes.Structure):
    """C struct: sattn_landmark_params_t"""
    _fields_ = [
        ("num_landmarks", ctypes.c_int),
        ("iters", ctypes.c_int),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

if _LIB_AVAILABLE:
    # Sliding window
    _lib.sattn_rvv_sliding_global.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnParamsT,
    ]
    _lib.sattn_rvv_sliding_global.restype = None

    _lib.sattn_rvv_sliding_global_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnParamsT,
    ]
    _lib.sattn_rvv_sliding_global_bf16.restype = None

    _lib.sattn_rvv_sliding_global_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_sliding_global_i8.restype = None

    _lib.sattn_rvv_sliding_global_i4.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_sliding_global_i4.restype = None

    # Block-topk / block_local_global
    _lib.sattn_rvv_block_topk.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnBlockTopKParamsT,
    ]
    _lib.sattn_rvv_block_topk.restype = None

    _lib.sattn_rvv_block_topk_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnBlockTopKParamsT,
    ]
    _lib.sattn_rvv_block_topk_bf16.restype = None

    _lib.sattn_rvv_block_topk_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnBlockTopKParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_block_topk_i8.restype = None

    _lib.sattn_rvv_block_topk_i4.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnBlockTopKParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_block_topk_i4.restype = None

    # N:M Structured
    _lib.sattn_rvv_nm_structured.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnNMStructuredParamsT,
    ]
    _lib.sattn_rvv_nm_structured.restype = None

    _lib.sattn_rvv_nm_structured_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnNMStructuredParamsT,
    ]
    _lib.sattn_rvv_nm_structured_bf16.restype = None

    _lib.sattn_rvv_nm_structured_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnNMStructuredParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_nm_structured_i8.restype = None

    _lib.sattn_rvv_nm_structured_i4.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnNMStructuredParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_nm_structured_i4.restype = None

    # LSH
    _lib.sattn_rvv_lsh.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLSHParamsT,
    ]
    _lib.sattn_rvv_lsh.restype = None

    _lib.sattn_rvv_lsh_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLSHParamsT,
    ]
    _lib.sattn_rvv_lsh_bf16.restype = None

    _lib.sattn_rvv_lsh_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLSHParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_lsh_i8.restype = None

    _lib.sattn_rvv_lsh_i4.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLSHParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_lsh_i4.restype = None

    # Landmark
    _lib.sattn_rvv_landmark.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLandmarkParamsT,
    ]
    _lib.sattn_rvv_landmark.restype = None

    _lib.sattn_rvv_landmark_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLandmarkParamsT,
    ]
    _lib.sattn_rvv_landmark_bf16.restype = None

    _lib.sattn_rvv_landmark_i8.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLandmarkParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_landmark_i8.restype = None

    _lib.sattn_rvv_landmark_i4.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        SattnShapeT, SattnLandmarkParamsT,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    _lib.sattn_rvv_landmark_i4.restype = None


# ============================================================================
# Python API
# ============================================================================

@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention operation"""
    pattern: SparseAttnPattern
    precision: Precision = Precision.FP32
    
    # Pattern-specific parameters
    window_size: Optional[int] = None  # sliding_window
    block_size: Optional[int] = None  # block_topk
    keep_ratio: Optional[float] = None  # block_topk
    global_tokens: Optional[int] = None  # block_topk
    N: Optional[int] = None  # nm_structured
    M: Optional[int] = None  # nm_structured
    num_hashes: Optional[int] = None  # lsh
    bucket_size: Optional[int] = None  # lsh
    num_landmarks: Optional[int] = None  # landmark
    
    # Quantization scales
    scale_q: float = 1.0 / 127.0
    scale_k: float = 1.0 / 127.0
    scale_v: float = 1.0 / 127.0


class SparseAttentionRVV:
    """
    Python interface to RVV sparse attention kernels.
    
    This class provides a clean Python API to call the optimized C implementations
    with Phase 1+2 optimizations (pre-quantization + vectorized dot product).
    
    Example:
        >>> import numpy as np
        >>> from sparse_attention_rvv import SparseAttentionRVV, SparseAttnPattern, Precision
        >>> 
        >>> # Create attention layer
        >>> attn = SparseAttentionRVV(
        ...     pattern=SparseAttnPattern.SLIDING_WINDOW,
        ...     precision=Precision.I8,
        ...     window_size=16
        ... )
        >>> 
        >>> # Run attention
        >>> Q = np.random.randn(1, 8, 128, 64).astype(np.float32)
        >>> K = np.random.randn(1, 8, 128, 64).astype(np.float32)
        >>> V = np.random.randn(1, 8, 128, 64).astype(np.float32)
        >>> O = attn(Q, K, V)
        >>> print(f"Output shape: {O.shape}")
    """
    
    def __init__(self, config: Union[SparseAttentionConfig, None] = None, **kwargs):
        """
        Initialize sparse attention layer.
        
        Args:
            config: SparseAttentionConfig object, or pass pattern/precision/params as kwargs
        """
        if not _LIB_AVAILABLE:
            raise RuntimeError("RVV library not available. Please build it first.")
        
        if config is None:
            config = SparseAttentionConfig(**kwargs)
        
        self.config = config
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        pattern = self.config.pattern
        
        if pattern == SparseAttnPattern.SLIDING_WINDOW:
            if self.config.window_size is None:
                raise ValueError("window_size required for sliding_window pattern")
        elif pattern == SparseAttnPattern.BLOCK_LOCAL_GLOBAL:
            if self.config.block_size is None or self.config.keep_ratio is None:
                raise ValueError("block_size and keep_ratio required for block_local_global")
        elif pattern == SparseAttnPattern.NM_STRUCTURED:
            if self.config.N is None or self.config.M is None:
                raise ValueError("N and M required for nm_structured pattern")
        elif pattern == SparseAttnPattern.LSH:
            if self.config.num_hashes is None or self.config.bucket_size is None:
                raise ValueError("num_hashes and bucket_size required for lsh pattern")
        elif pattern == SparseAttnPattern.LANDMARK:
            if self.config.num_landmarks is None:
                raise ValueError("num_landmarks required for landmark pattern")
    
    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Forward pass of sparse attention.
        
        Args:
            Q: Query tensor [B, H, L, D] (float32)
            K: Key tensor [B, H, L, D] (float32)
            V: Value tensor [B, H, L, D] (float32)
        
        Returns:
            O: Output tensor [B, H, L, D] (float32)
        """
        return self.forward(Q, K, V)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Forward pass of sparse attention.
        
        Args:
            Q: Query tensor [B, H, L, D] (float32)
            K: Key tensor [B, H, L, D] (float32)
            V: Value tensor [B, H, L, D] (float32)
        
        Returns:
            O: Output tensor [B, H, L, D] (float32)
        """
        # Validate input shapes
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError(f"Q, K, V must have same shape, got {Q.shape}, {K.shape}, {V.shape}")
        
        if len(Q.shape) != 4:
            raise ValueError(f"Expected 4D tensors [B, H, L, D], got shape {Q.shape}")
        
        # Ensure float32 and contiguous
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)
        
        # Allocate output
        O = np.zeros_like(Q, dtype=np.float32)
        
        # Get shape
        B, H, L, D = Q.shape
        shape = SattnShapeT(B=B, H=H, L=L, D=D)
        
        # Dispatch to appropriate backend
        if self.config.pattern == SparseAttnPattern.SLIDING_WINDOW:
            self._forward_sliding_window(Q, K, V, O, shape)
        elif self.config.pattern == SparseAttnPattern.BLOCK_LOCAL_GLOBAL:
            self._forward_block_topk(Q, K, V, O, shape)
        elif self.config.pattern == SparseAttnPattern.NM_STRUCTURED:
            self._forward_nm_structured(Q, K, V, O, shape)
        elif self.config.pattern == SparseAttnPattern.LSH:
            self._forward_lsh(Q, K, V, O, shape)
        elif self.config.pattern == SparseAttnPattern.LANDMARK:
            self._forward_landmark(Q, K, V, O, shape)
        else:
            raise ValueError(f"Unknown pattern: {self.config.pattern}")
        
        return O
    
    def _forward_sliding_window(self, Q, K, V, O, shape):
        """Call sliding_window C function"""
        params = SattnParamsT(
            window_size=self.config.window_size,
            block_size=0,
            dilation=1,
            wrap=0,
        )
        
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if self.config.precision == Precision.FP32:
            _lib.sattn_rvv_sliding_global(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.BF16:
            _lib.sattn_rvv_sliding_global_bf16(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.I8:
            _lib.sattn_rvv_sliding_global_i8(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        elif self.config.precision == Precision.I4:
            _lib.sattn_rvv_sliding_global_i4(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")
    
    def _forward_block_topk(self, Q, K, V, O, shape):
        """Call block_topk C function"""
        params = SattnBlockTopKParamsT(
            block_size=self.config.block_size or 64,
            keep_ratio=self.config.keep_ratio or 0.12,
            global_tokens=self.config.global_tokens or 0,
            gqa_group_size=1,  # Default: no grouped query attention
            comp_block_size=0,  # Default: no compression
        )
        
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if self.config.precision == Precision.FP32:
            _lib.sattn_rvv_block_topk(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.BF16:
            _lib.sattn_rvv_block_topk_bf16(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.I8:
            _lib.sattn_rvv_block_topk_i8(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        elif self.config.precision == Precision.I4:
            _lib.sattn_rvv_block_topk_i4(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")
    
    def _forward_nm_structured(self, Q, K, V, O, shape):
        """Call nm_structured C function"""
        params = SattnNMStructuredParamsT(
            n=self.config.N or 2,
            m=self.config.M or 4,
        )
        
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if self.config.precision == Precision.FP32:
            _lib.sattn_rvv_nm_structured(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.BF16:
            _lib.sattn_rvv_nm_structured_bf16(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.I8:
            _lib.sattn_rvv_nm_structured_i8(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        elif self.config.precision == Precision.I4:
            _lib.sattn_rvv_nm_structured_i4(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")
    
    def _forward_lsh(self, Q, K, V, O, shape):
        """Call lsh C function"""
        # Use num_hashes if provided, otherwise bucket_size, else default to 4
        buckets = self.config.num_hashes or self.config.bucket_size or 4
        params = SattnLSHParamsT(
            buckets=buckets,
        )
        
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if self.config.precision == Precision.FP32:
            _lib.sattn_rvv_lsh(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.BF16:
            _lib.sattn_rvv_lsh_bf16(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.I8:
            _lib.sattn_rvv_lsh_i8(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        elif self.config.precision == Precision.I4:
            _lib.sattn_rvv_lsh_i4(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")
    
    def _forward_landmark(self, Q, K, V, O, shape):
        """Call landmark C function"""
        params = SattnLandmarkParamsT(
            num_landmarks=self.config.num_landmarks or 16,
            iters=0,  # Default: no refinement
        )
        
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if self.config.precision == Precision.FP32:
            _lib.sattn_rvv_landmark(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.BF16:
            _lib.sattn_rvv_landmark_bf16(Q_ptr, K_ptr, V_ptr, O_ptr, shape, params)
        elif self.config.precision == Precision.I8:
            _lib.sattn_rvv_landmark_i8(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        elif self.config.precision == Precision.I4:
            _lib.sattn_rvv_landmark_i4(
                Q_ptr, K_ptr, V_ptr, O_ptr, shape, params,
                self.config.scale_q, self.config.scale_k, self.config.scale_v
            )
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")


# ============================================================================
# Convenience Functions
# ============================================================================

def sparse_attention_rvv(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    pattern: Union[SparseAttnPattern, str],
    precision: Union[Precision, str] = "fp32",
    **kwargs
) -> np.ndarray:
    """
    Convenience function for sparse attention.
    
    Args:
        Q, K, V: Input tensors [B, H, L, D]
        pattern: Sparse attention pattern
        precision: Precision level
        **kwargs: Pattern-specific parameters
    
    Returns:
        O: Output tensor [B, H, L, D]
    
    Example:
        >>> O = sparse_attention_rvv(Q, K, V, "sliding_window", "i8", window_size=16)
    """
    if isinstance(pattern, str):
        pattern = SparseAttnPattern(pattern)
    if isinstance(precision, str):
        precision = Precision(precision)
    
    attn = SparseAttentionRVV(
        pattern=pattern,
        precision=precision,
        **kwargs
    )
    
    return attn(Q, K, V)


if __name__ == "__main__":
    # Example usage
    if _LIB_AVAILABLE:
        print("Testing RVV Python bindings...")
        
        # Create random inputs
        B, H, L, D = 1, 2, 32, 16
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        # Test different precisions
        for precision in [Precision.FP32, Precision.BF16, Precision.I8, Precision.I4]:
            print(f"\nTesting {precision.value}...")
            attn = SparseAttentionRVV(
                pattern=SparseAttnPattern.SLIDING_WINDOW,
                precision=precision,
                window_size=8
            )
            O = attn(Q, K, V)
            print(f"  Output shape: {O.shape}")
            print(f"  Output range: [{O.min():.4f}, {O.max():.4f}]")
            print(f"  ✅ Success!")
    else:
        print("RVV library not available. Skipping tests.")

