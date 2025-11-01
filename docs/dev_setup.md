# Developer Setup

Two supported flows: local Python venv or Docker (CUDA-enabled).

## 1) Local Python (venv)
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Optional GPU
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton
pytest -q
```

## 2) Docker (CUDA)
Requires NVIDIA drivers and `nvidia-container-toolkit`.

Build:
```bash
docker build -f docker/Dockerfile -t sattn:cuda .
```

Run (GPU):
```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace sattn:cuda bash
pytest -q
```

Notes:
- Triton/PyTorch versions in the Dockerfile are pinned for CUDA 12.1.
- For RVV builds, the container includes `riscv64-linux-gnu-gcc` and `qemu-riscv64`.
- Cross-build and run RVV on QEMU:
  ```bash
  python scripts/build_and_run_rvv_qemu.py
  ```
  This uses `backends/rvv/toolchains/linux-gnu-rvv.cmake` and runs a compare test under QEMU user-mode.
