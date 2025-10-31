# CUDA 12.1 base with Ubuntu 22.04
# Switch to Ubuntu 24.04 base to access MLIR dev packages
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential make cmake pkg-config ninja-build \
    wget gnupg ca-certificates lsb-release \
    libedit-dev zlib1g-dev libzstd-dev libcurl4-openssl-dev \
    verilator \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python venv for pip installs (PEP 668 compliant)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Fetch official LLVM 18.1.8 binaries with MLIR for Ubuntu 24.04
RUN set -eux; \
    # Try prebuilt LLVM first (Ubuntu 24.04), then fallback to apt.llvm.org
    LLVM_PKG="clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-24.04.tar.xz"; \
    if wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/${LLVM_PKG}; then \
      mkdir -p /opt && tar -xf ${LLVM_PKG} -C /opt && rm -f ${LLVM_PKG}; \
      ln -sfn /opt/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-24.04 /opt/llvm-18; \
    else \
      echo "[info] Falling back to apt.llvm.org"; \
      echo "deb http://apt.llvm.org/$(. /etc/os-release && echo ${UBUNTU_CODENAME})/ llvm-toolchain-$(. /etc/os-release && echo ${UBUNTU_CODENAME})-18 main" \
        | tee /etc/apt/sources.list.d/llvm-18.list; \
      wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -; \
      apt-get update; \
      apt-get install -y --no-install-recommends clang-18 lld-18 llvm-18-dev mlir-18-tools libmlir-18-dev; \
      ln -sfn /usr/lib/llvm-18 /opt/llvm-18; \
    fi

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command: interactive shell (compose overrides to sleep for dev)
CMD ["bash"]


