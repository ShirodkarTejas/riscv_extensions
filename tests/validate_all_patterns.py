#!/usr/bin/env python3
"""
Validate all sparse attention patterns using proven subprocess+QEMU approach.

This script tests all 5 patterns × 4 precisions = 20 configurations
by calling the sattn_rvv_runner executable via QEMU.
"""

import subprocess
import json
import sys
from pathlib import Path

# Configuration
PATTERNS = ["sliding_window", "block_local_global", "nm_structured", "lsh", "landmark"]
PRECISIONS = ["fp32", "bf16", "i8", "i4"]
QEMU_CMD = "qemu-riscv64"
QEMU_CPU = "rv64,v=true,vlen=256"
RUNNER = "/workspace/backends/rvv/build/sattn_rvv_runner"

# Test configuration
L = 32  # Small for fast testing
D = 16

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_pattern(pattern: str, precision: str, L: int, D: int):
    """Run a single pattern configuration via QEMU"""
    cmd = [
        QEMU_CMD,
        "-L", "/usr/riscv64-linux-gnu",  # Sysroot for dynamic linker
        "-cpu", QEMU_CPU,
        RUNNER,
        "--spec", pattern,
        "--L", str(L),
        "--D", str(D),
        "--precision", precision,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/workspace"
        )
        
        if result.returncode != 0:
            return None, f"Exit code: {result.returncode}, stderr: {result.stderr[:200]}"
        
        # Try to parse output (should contain cycles and memory info)
        output = result.stdout.strip()
        if not output:
            return None, "No output"
        
        # Look for key metrics
        has_cycles = "cycles" in output.lower() or "cycle" in output.lower()
        has_memory = "memory" in output.lower() or "bytes" in output.lower()
        
        return {
            "success": True,
            "has_cycles": has_cycles,
            "has_memory": has_memory,
            "output_lines": len(output.split('\n'))
        }, None
        
    except subprocess.TimeoutExpired:
        return None, "Timeout (30s)"
    except Exception as e:
        return None, str(e)


def main():
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}Validating All Sparse Attention Patterns{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"Problem size: L={L}, D={D}")
    print(f"Total configurations: {len(PATTERNS)} patterns × {len(PRECISIONS)} precisions = {len(PATTERNS) * len(PRECISIONS)}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    results = {}
    total = 0
    passed = 0
    failed = 0
    
    for pattern in PATTERNS:
        print(f"\n{YELLOW}### {pattern} ###{RESET}")
        results[pattern] = {}
        
        for precision in PRECISIONS:
            total += 1
            config_name = f"{pattern}/{precision}"
            
            print(f"  [{total:2d}/{len(PATTERNS)*len(PRECISIONS)}] Testing {precision:4s} ... ", end="", flush=True)
            
            result, error = run_pattern(pattern, precision, L, D)
            
            if result:
                passed += 1
                print(f"{GREEN}✅ PASS{RESET}")
                results[pattern][precision] = "PASS"
            else:
                failed += 1
                print(f"{RED}❌ FAIL{RESET} - {error[:60]}")
                results[pattern][precision] = f"FAIL: {error}"
    
    # Summary
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}SUMMARY{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"Total configurations tested: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    if failed > 0:
        print(f"{RED}Failed: {failed}{RESET}")
    else:
        print(f"Failed: 0")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    # Detailed results table
    print(f"\n{BLUE}Detailed Results:{RESET}")
    print(f"{'Pattern':<20} {'FP32':<8} {'BF16':<8} {'I8':<8} {'I4':<8}")
    print("-" * 60)
    
    for pattern in PATTERNS:
        row = [pattern]
        for precision in PRECISIONS:
            status = results[pattern].get(precision, "SKIP")
            if status == "PASS":
                row.append(f"{GREEN}✅{RESET}")
            else:
                row.append(f"{RED}❌{RESET}")
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15}")
    
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    # Final verdict
    if failed == 0:
        print(f"{GREEN}🎉 ALL TESTS PASSED! All {passed} configurations working!{RESET}")
        return 0
    else:
        print(f"{YELLOW}⚠️  Some tests failed. Check the output above for details.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

