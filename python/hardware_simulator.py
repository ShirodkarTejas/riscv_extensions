#!/usr/bin/env python3
"""
Python Interface to Hardware Simulator (Verilator)

Provides a Python-friendly interface to run sparse attention on
cycle-accurate hardware simulation.
"""

import subprocess
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile

class HardwareSimulator:
    """
    Interface to Verilator-simulated sparse attention accelerator
    
    Usage:
        hw_sim = HardwareSimulator('path/to/simulator-SattnRocketConfig')
        O, metrics = hw_sim.run(Q, K, V, pattern='sliding_window', precision='fp32')
    """
    
    def __init__(self, verilator_binary: str, timeout: int = 300):
        """
        Initialize hardware simulator
        
        Args:
            verilator_binary: Path to Verilator simulator executable
            timeout: Maximum simulation time in seconds
        """
        self.verilator_bin = Path(verilator_binary)
        self.timeout = timeout
        
        if not self.verilator_bin.exists():
            raise FileNotFoundError(f"Verilator binary not found: {verilator_binary}")
    
    def run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        pattern: str,
        precision: str = 'fp32',
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run sparse attention on hardware simulator
        
        Args:
            Q: Query tensor [B, H, L, D]
            K: Key tensor [B, H, L, D]
            V: Value tensor [B, H, L, D]
            pattern: Sparse attention pattern
            precision: Precision level (fp32/bf16/i8/i4)
            **kwargs: Pattern-specific parameters
        
        Returns:
            (output tensor, performance metrics dict)
        """
        
        # Validate inputs
        assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
        assert len(Q.shape) == 4, "Tensors must be 4D [B, H, L, D]"
        
        B, H, L, D = Q.shape
        
        # Create temporary directory for I/O
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write inputs
            np.save(tmpdir / 'Q.npy', Q.astype(np.float32))
            np.save(tmpdir / 'K.npy', K.astype(np.float32))
            np.save(tmpdir / 'V.npy', V.astype(np.float32))
            
            # Build command
            cmd = [
                str(self.verilator_bin),
                f'+pattern={pattern}',
                f'+precision={precision}',
                f'+Q={tmpdir / "Q.npy"}',
                f'+K={tmpdir / "K.npy"}',
                f'+V={tmpdir / "V.npy"}',
                f'+O={tmpdir / "O.npy"}',
                f'+B={B}',
                f'+H={H}',
                f'+L={L}',
                f'+D={D}',
            ]
            
            # Add pattern-specific parameters
            if pattern == 'sliding_window':
                cmd.append(f'+window_size={kwargs.get("window_size", 16)}')
            elif pattern == 'block_local_global':
                cmd.append(f'+block_size={kwargs.get("block_size", 16)}')
                cmd.append(f'+keep_ratio={kwargs.get("keep_ratio", 0.10)}')
                cmd.append(f'+global_tokens={kwargs.get("global_tokens", 4)}')
            elif pattern == 'nm_structured':
                cmd.append(f'+nm_n={kwargs.get("nm_n", 2)}')
                cmd.append(f'+nm_m={kwargs.get("nm_m", 4)}')
            elif pattern == 'lsh':
                cmd.append(f'+buckets={kwargs.get("buckets", 8)}')
            elif pattern == 'landmark':
                cmd.append(f'+num_landmarks={kwargs.get("num_landmarks", 16)}')
            
            # Run simulation
            print(f"Running hardware simulation (pattern={pattern}, precision={precision})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Hardware simulation failed:\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )
            
            # Parse performance metrics
            metrics = self._parse_output(result.stdout)
            
            # Read output
            O = np.load(tmpdir / 'O.npy')
            
            return O, metrics
    
    def _parse_output(self, output: str) -> Dict:
        """Parse Verilator output for performance metrics"""
        metrics = {
            'cycles': 0,
            'mem_reads': 0,
            'mem_writes': 0,
            'sim_time_sec': 0.0,
        }
        
        for line in output.split('\n'):
            line = line.strip()
            if 'CYCLES:' in line:
                metrics['cycles'] = int(line.split(':')[1].strip())
            elif 'MEM_READS:' in line:
                metrics['mem_reads'] = int(line.split(':')[1].strip())
            elif 'MEM_WRITES:' in line:
                metrics['mem_writes'] = int(line.split(':')[1].strip())
            elif 'SIM_TIME:' in line:
                metrics['sim_time_sec'] = float(line.split(':')[1].strip())
        
        return metrics

    def validate_vs_software(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        pattern: str,
        precision: str = 'fp32',
        reference_output: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Run hardware simulation and validate against software reference
        
        Args:
            Q, K, V: Input tensors
            pattern, precision: Configuration
            reference_output: Software reference output (optional)
            **kwargs: Pattern parameters
        
        Returns:
            dict with hardware output, metrics, and validation results
        """
        
        # Run on hardware
        O_hw, hw_metrics = self.run(Q, K, V, pattern, precision, **kwargs)
        
        results = {
            'hardware_output': O_hw,
            'hardware_metrics': hw_metrics,
        }
        
        # Validate if reference provided
        if reference_output is not None:
            mae = np.mean(np.abs(O_hw - reference_output))
            max_diff = np.max(np.abs(O_hw - reference_output))
            
            results['validation'] = {
                'mae': float(mae),
                'max_diff': float(max_diff),
                'passed': mae < 1e-3,  # Tolerance
            }
            
            print(f"Validation: MAE={mae:.6f}, Max Diff={max_diff:.6f}")
            print(f"  {'✅ PASS' if results['validation']['passed'] else '❌ FAIL'}")
        
        return results


# =============================================================================
# Helper Functions
# =============================================================================

def compare_hardware_software(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    pattern: str,
    precision: str = 'fp32',
    hw_simulator_path: str = None,
    **kwargs
):
    """
    Run same configuration on hardware and software, compare results
    
    This is the main validation function for hardware correctness.
    """
    
    # Import software backend
    from sparse_attention_rvv import sparse_attention_rvv
    from dense_attention_reference import dense_attention_reference
    
    print(f"{'='*70}")
    print(f"Hardware vs Software Comparison")
    print(f"  Pattern: {pattern}")
    print(f"  Precision: {precision}")
    print(f"  Shape: {Q.shape}")
    print(f"{'='*70}\n")
    
    # 1. Run on hardware
    if hw_simulator_path:
        hw_sim = HardwareSimulator(hw_simulator_path)
        O_hw, hw_metrics = hw_sim.run(Q, K, V, pattern, precision, **kwargs)
        print(f"✅ Hardware simulation complete")
        print(f"   Cycles: {hw_metrics['cycles']}")
        print(f"   Memory ops: {hw_metrics['mem_reads'] + hw_metrics['mem_writes']}")
    else:
        print("⚠️  No hardware simulator specified, skipping hardware run")
        O_hw = None
        hw_metrics = None
    
    # 2. Run on software (Phase 1)
    print(f"\nRunning software reference...")
    O_sw = sparse_attention_rvv(Q, K, V, pattern=pattern, precision=precision, **kwargs)
    print(f"✅ Software execution complete")
    
    # 3. Run dense reference
    print(f"\nRunning dense reference...")
    O_ref = dense_attention_reference(Q, K, V)
    print(f"✅ Dense reference complete")
    
    # 4. Compare
    print(f"\n{'='*70}")
    print(f"Comparison Results")
    print(f"{'='*70}")
    
    mae_sw_ref = np.mean(np.abs(O_sw - O_ref))
    print(f"Software vs Dense Reference:")
    print(f"  MAE: {mae_sw_ref:.6f}")
    
    if O_hw is not None:
        mae_hw_ref = np.mean(np.abs(O_hw - O_ref))
        mae_hw_sw = np.mean(np.abs(O_hw - O_sw))
        
        print(f"\nHardware vs Dense Reference:")
        print(f"  MAE: {mae_hw_ref:.6f}")
        
        print(f"\nHardware vs Software:")
        print(f"  MAE: {mae_hw_sw:.6f}")
        print(f"  Status: {'✅ PASS' if mae_hw_sw < 1e-3 else '❌ FAIL'}")
        
        return {
            'O_hw': O_hw,
            'O_sw': O_sw,
            'O_ref': O_ref,
            'mae_hw_sw': float(mae_hw_sw),
            'mae_hw_ref': float(mae_hw_ref),
            'mae_sw_ref': float(mae_sw_ref),
            'hw_metrics': hw_metrics,
        }
    else:
        return {
            'O_sw': O_sw,
            'O_ref': O_ref,
            'mae_sw_ref': float(mae_sw_ref),
        }


if __name__ == "__main__":
    # Example usage
    print("Hardware Simulator Interface Test")
    print("="*70)
    
    # Generate test data
    B, H, L, D = 1, 2, 32, 16
    Q = np.random.randn(B, H, L, D).astype(np.float32)
    K = np.random.randn(B, H, L, D).astype(np.float32)
    V = np.random.randn(B, H, L, D).astype(np.float32)
    
    # Run comparison (would need actual simulator)
    # results = compare_hardware_software(
    #     Q, K, V, 
    #     pattern='sliding_window',
    #     precision='fp32',
    #     hw_simulator_path='chipyard/sims/verilator/simulator-SattnRocketConfig',
    #     window_size=16
    # )
    
    print("\n✅ Hardware simulator interface ready!")
    print("   To use: Set up Chipyard and build Verilator simulator")
    print("   Then run: python hardware_simulator.py")

