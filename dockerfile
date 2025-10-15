# Triton 24.05 → CUDA 12.1
FROM nvcr.io/nvidia/tritonserver:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# System deps (opencv needs libgl; pycocotools needs build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git python3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Python toolchain
RUN python3 -m pip install -U pip setuptools wheel

# Torch stack pinned to CUDA 12.1 (matches Triton 24.05)
# torchvision version must match torch. 2.3.1 ↔ 0.18.1
RUN pip install --retries 5 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

RUN pip install --retries 5 \
    numpy>=1.24 \
    transformers==4.44.2 \
    timm==1.0.7 \
    opencv-python==4.10.0.84 \
    supervision>=0.22.0 \
    pycocotools==2.0.8 \
    addict==2.4.0 \
    yapf==0.40.2

RUN pip install --retries 5 \
    git+https://github.com/IDEA-Research/GroundingDINO.git

COPY groundingdino_py /models/groundingdino_py
WORKDIR /workspace
EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver","--model-repository=/models"]
