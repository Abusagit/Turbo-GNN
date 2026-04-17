# =============================================================================
# Turbo-GNN — local dev image
#
# Build: docker build --load -t turbo-gnn:dev .
# Run:   docker run --gpus all -it turbo-gnn:dev bash
# Run with live code: docker run --gpus all -it -v $(pwd):/workspace/turbo-gnn turbo-gnn:dev bash
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.09-py3

LABEL version="1.0"
LABEL description="Turbo-GNN research image — CUDA kernels, PyG, Triton"


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        vim \
        nano \
        make \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# CUDA env
# --------------------------------------------------------------------------- #
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"


RUN pip uninstall -y torch torchvision torchaudio || true


WORKDIR /workspace/turbo-gnn
COPY . .


RUN make _install-torch-dev \
        PYTHON_BIN=$(which python3) \
        VENV_DIR="" \
        PIP="pip3" \
        PYTHON="python3" \
        CUDA_HOME=${CUDA_HOME}

RUN make _install-dev \
        PYTHON_BIN=$(which python3) \
        VENV_DIR="" \
        PIP="pip3" \
        PYTHON="python3" \
        CUDA_HOME=${CUDA_HOME}


ENV DGLBACKEND="pytorch"
ENV PYTHONPATH="/workspace/turbo-gnn:${PYTHONPATH}"

CMD ["bash"]
