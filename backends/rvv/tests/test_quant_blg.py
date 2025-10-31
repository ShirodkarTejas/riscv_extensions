import os
import subprocess


def run(cmd):
    out = subprocess.check_output(cmd, cwd=os.path.join(os.getcwd(), "build", "backends", "rvv"))
    return out.decode("utf-8", errors="ignore")


def parse_checksum(s):
    for line in s.splitlines():
        if line.startswith("spec="):
            for p in line.split():
                if p.startswith("checksum="):
                    return float(p.split("=")[1])
    raise AssertionError("checksum not found")


def test_quant_blg_bf16():
    base = run(["./sattn_rvv_runner", "--spec", "block_local_global", "--L", "64", "--D", "32", "--block_size", "16", "--keep_x1000", "120", "--global_tokens", "4"]) 
    qbf16 = run(["./sattn_rvv_runner", "--spec", "block_local_global", "--L", "64", "--D", "32", "--block_size", "16", "--keep_x1000", "120", "--global_tokens", "4", "--precision", "bf16"]) 
    c0 = parse_checksum(base)
    c1 = parse_checksum(qbf16)
    assert abs(c0 - c1) / (abs(c0) + 1e-9) < 1e-2


def test_quant_blg_i8_i4():
    base = run(["./sattn_rvv_runner", "--spec", "block_local_global", "--L", "64", "--D", "32", "--block_size", "16", "--keep_x1000", "120", "--global_tokens", "4"]) 
    qi8 = run(["./sattn_rvv_runner", "--spec", "block_local_global", "--L", "64", "--D", "32", "--block_size", "16", "--keep_x1000", "120", "--global_tokens", "4", "--precision", "i8", "--scale_q_x1000", "50", "--scale_k_x1000", "50", "--scale_v_x1000", "50"]) 
    qi4 = run(["./sattn_rvv_runner", "--spec", "block_local_global", "--L", "64", "--D", "32", "--block_size", "16", "--keep_x1000", "120", "--global_tokens", "4", "--precision", "i4", "--scale_q_x1000", "125", "--scale_k_x1000", "125", "--scale_v_x1000", "125"]) 
    c0 = parse_checksum(base)
    c8 = parse_checksum(qi8)
    c4 = parse_checksum(qi4)
    assert abs(c0 - c8) / (abs(c0) + 1e-9) < 5e-2
    assert abs(c0 - c4) / (abs(c0) + 1e-9) < 2e-1


