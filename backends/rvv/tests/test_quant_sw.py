import os
import subprocess
import sys


def run(cmd):
    print("RUN:", " ".join(cmd))
    out = subprocess.check_output(cmd, cwd=os.path.join(os.getcwd(), "build", "backends", "rvv"))
    s = out.decode("utf-8", errors="ignore")
    print(s)
    return s


def parse_checksum(s):
    # line: spec=... checksum=... rvv_bytes_read=...
    for line in s.splitlines():
        if line.startswith("spec="):
            parts = line.split()
            for p in parts:
                if p.startswith("checksum="):
                    return float(p.split("=")[1])
    raise AssertionError("checksum not found")


def test_quant_sliding_window_fp32_vs_bf16():
    base = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8"]) 
    qbf16 = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8", "--precision", "bf16"]) 
    c0 = parse_checksum(base)
    c1 = parse_checksum(qbf16)
    # bf16 error tolerance (loose)
    rel = abs(c0 - c1) / (abs(c0) + 1e-9)
    assert rel < 1e-2


def test_quant_sliding_window_i8():
    base = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8"]) 
    qi8 = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8", "--precision", "i8", "--scale_q_x1000", "50", "--scale_k_x1000", "50", "--scale_v_x1000", "50"]) 
    c0 = parse_checksum(base)
    c1 = parse_checksum(qi8)
    # int8 error tolerance (looser)
    rel = abs(c0 - c1) / (abs(c0) + 1e-9)
    assert rel < 5e-2


def test_quant_sliding_window_i4():
    base = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8"]) 
    qi4 = run(["./sattn_rvv_runner", "--spec", "sliding_window", "--L", "64", "--D", "32", "--window", "8", "--precision", "i4", "--scale_q_x1000", "125", "--scale_k_x1000", "125", "--scale_v_x1000", "125"]) 
    c0 = parse_checksum(base)
    c1 = parse_checksum(qi4)
    rel = abs(c0 - c1) / (abs(c0) + 1e-9)
    assert rel < 2.0e-1


