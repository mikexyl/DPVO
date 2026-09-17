# Validated target: Orin NX, JetPack 7.2 / L4T R39.2 (CUDA 13.2).
ARG DPVO_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${DPVO_BASE_IMAGE}
ARG MAX_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=8.7 \
    MAX_JOBS=${MAX_JOBS} \
    FORCE_CUDA=1 \
    CONDA_PREFIX=/usr \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir yacs einops evo numba \
    opencv-python-headless plyfile matplotlib onnx onnxscript \
    && python -m pip install --no-cache-dir --no-deps pypose==0.9.5 \
    && python -m pip install --no-cache-dir --no-build-isolation torch-scatter==2.1.2
WORKDIR /opt/dpvo
COPY setup.py ./
COPY dpvo ./dpvo
RUN python -m pip install --no-build-isolation --no-deps .
COPY config ./config
COPY calib ./calib
COPY deploy/jetson ./deploy/jetson
COPY demo.py ./
# Some DPVO directories are namespace packages omitted by setup.py's wheel.
ENV PYTHONPATH=/opt/dpvo
CMD ["bash"]
