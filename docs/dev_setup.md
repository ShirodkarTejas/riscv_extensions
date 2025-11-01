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
docker build -f docker/Dockerfile -t sattn/rvv-dev:latest .
```

Run (GPU):
```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace sattn/rvv-dev:latest bash
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

 - One-shot QEMU run via docker (no compose, no PowerShell):
   ```bash
   docker run --rm -v $PWD:/workspace -w /workspace sattn/rvv-dev:latest \
     bash -lc 'cmake --build build/rvv-riscv64 -v && \
               qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen=128,elen=64 \
               build/rvv-riscv64/sattn_rvv_compare_sw'
   ```
   Replace the executable with the runner to try other specs:
   ```bash
   docker run --rm -v $PWD:/workspace -w /workspace sattn/rvv-dev:latest \
     bash -lc 'cmake --build build/rvv-riscv64 -v && \
               qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen=128,elen=64 \
               build/rvv-riscv64/sattn_rvv_runner --spec sliding_window --L 128 --D 32 --window 8'
   ```