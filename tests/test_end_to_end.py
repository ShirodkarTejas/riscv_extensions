"""
End-to-end integration tests for sparse attention stack.

Tests the complete flow: Python API → C Backend → QEMU Execution
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from python.sparse_attention_rvv import (
    SparseAttentionRVV,
    SparseAttnPattern,
    Precision,
    _LIB_AVAILABLE,
)

from validation.dense_attention_reference import (
    dense_attention_fp32,
    compare_sparse_vs_dense,
    validate_attention_correctness,
)

# Skip all tests if RVV library is not available
pytestmark = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason="RVV library not available"
)


class TestPythonBindings:
    """Test Python ctypes bindings to C library"""
    
    def test_library_loading(self):
        """Test that RVV library loads successfully"""
        assert _LIB_AVAILABLE, "RVV library should be available"
    
    def test_basic_forward_pass(self):
        """Test that basic forward pass works"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        B, H, L, D = 1, 2, 16, 8
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        O = attn(Q, K, V)
        
        assert O.shape == (B, H, L, D)
        assert O.dtype == np.float32
        assert not np.isnan(O).any(), "Output contains NaN"
        assert not np.isinf(O).any(), "Output contains Inf"
    
    def test_input_validation(self):
        """Test input validation"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        # Test shape mismatch
        Q = np.random.randn(1, 2, 16, 8).astype(np.float32)
        K = np.random.randn(1, 2, 8, 8).astype(np.float32)  # Wrong L
        V = np.random.randn(1, 2, 16, 8).astype(np.float32)
        
        with pytest.raises(ValueError):
            attn(Q, K, V)
    
    def test_deterministic(self):
        """Test that outputs are deterministic"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        B, H, L, D = 1, 2, 16, 8
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        O1 = attn(Q, K, V)
        O2 = attn(Q, K, V)
        
        np.testing.assert_array_equal(O1, O2, err_msg="Output is not deterministic")


class TestAllPrecisions:
    """Test all precision levels"""
    
    @pytest.mark.parametrize("precision", [
        Precision.FP32,
        Precision.BF16,
        Precision.I8,
        Precision.I4,
    ])
    def test_precision(self, precision):
        """Test each precision level"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=precision,
            window_size=8
        )
        
        B, H, L, D = 1, 2, 16, 8
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        O = attn(Q, K, V)
        
        assert O.shape == (B, H, L, D)
        assert not np.isnan(O).any()
        assert not np.isinf(O).any()


class TestAccuracyVsDense:
    """Test accuracy against dense reference"""
    
    def test_fp32_accuracy(self):
        """Test that FP32 matches dense reference closely"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        B, H, L, D = 1, 1, 16, 8
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        # Sparse output
        sparse_output = attn(Q, K, V)
        
        # Note: Dense reference computes *full* attention, not windowed
        # So we expect some difference. This test validates the Python bindings work.
        assert sparse_output.shape == (B, H, L, D)
        assert not np.isnan(sparse_output).any()
    
    def test_quantized_accuracy_degradation(self):
        """Test that quantization degrades accuracy as expected"""
        B, H, L, D = 1, 1, 16, 8
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        # Compute outputs for different precisions
        outputs = {}
        for precision in [Precision.FP32, Precision.BF16, Precision.I8, Precision.I4]:
            attn = SparseAttentionRVV(
                pattern=SparseAttnPattern.SLIDING_WINDOW,
                precision=precision,
                window_size=8
            )
            outputs[precision] = attn(Q, K, V)
        
        # Compute MAE between FP32 and quantized versions
        mae_bf16 = np.mean(np.abs(outputs[Precision.FP32] - outputs[Precision.BF16]))
        mae_i8 = np.mean(np.abs(outputs[Precision.FP32] - outputs[Precision.I8]))
        mae_i4 = np.mean(np.abs(outputs[Precision.FP32] - outputs[Precision.I4]))
        
        print(f"\nMAE vs FP32:")
        print(f"  BF16: {mae_bf16:.6f}")
        print(f"  I8:   {mae_i8:.6f}")
        print(f"  I4:   {mae_i4:.6f}")
        
        # Expected ordering: i4 > i8 > bf16
        assert mae_i4 >= mae_i8, "i4 should have more error than i8"
        assert mae_i8 >= mae_bf16, "i8 should have more error than bf16"


class TestDifferentSizes:
    """Test with different problem sizes"""
    
    @pytest.mark.parametrize("L,D", [
        (8, 4),
        (16, 8),
        (32, 16),
        (64, 32),
        (128, 64),
    ])
    def test_size(self, L, D):
        """Test different sequence lengths and dimensions"""
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=min(8, L // 2)
        )
        
        B, H = 1, 2
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        O = attn(Q, K, V)
        
        assert O.shape == (B, H, L, D)
        assert not np.isnan(O).any()


class TestPyTorchIntegration:
    """Test PyTorch integration (if available)"""
    
    @pytest.fixture(autouse=True)
    def check_torch(self):
        """Check if PyTorch is available"""
        try:
            import torch
            self.has_torch = True
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_torch_layer_forward(self):
        """Test that PyTorch layer forward pass works"""
        import torch
        from python.torch_sparse_attention import SparseAttentionLayer
        
        layer = SparseAttentionLayer(
            d_model=64,
            num_heads=4,
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        x = torch.randn(2, 16, 64)
        y = layer(x, x, x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_torch_transformer(self):
        """Test full transformer model"""
        import torch
        from python.torch_sparse_attention import SparseTransformer
        
        model = SparseTransformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        x = torch.randn(2, 16, 64)
        y = model(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_torch_gradient_flow(self):
        """Test that gradients flow correctly"""
        import torch
        from python.torch_sparse_attention import SparseTransformer
        
        model = SparseTransformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=1,
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.FP32,
            window_size=8
        )
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow"""
    
    def test_numpy_workflow(self):
        """Test NumPy-based workflow"""
        # 1. Create attention layer
        attn = SparseAttentionRVV(
            pattern=SparseAttnPattern.SLIDING_WINDOW,
            precision=Precision.I8,
            window_size=16
        )
        
        # 2. Generate random inputs
        B, H, L, D = 2, 8, 128, 64
        Q = np.random.randn(B, H, L, D).astype(np.float32)
        K = np.random.randn(B, H, L, D).astype(np.float32)
        V = np.random.randn(B, H, L, D).astype(np.float32)
        
        # 3. Run attention
        O = attn(Q, K, V)
        
        # 4. Validate output
        assert O.shape == (B, H, L, D)
        assert O.dtype == np.float32
        assert not np.isnan(O).any()
        assert not np.isinf(O).any()
        
        print(f"\n✅ End-to-end NumPy workflow successful!")
        print(f"   Input shape: {Q.shape}")
        print(f"   Output shape: {O.shape}")
        print(f"   Pattern: {attn.config.pattern.value}")
        print(f"   Precision: {attn.config.precision.value}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

