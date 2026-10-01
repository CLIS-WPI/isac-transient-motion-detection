# mdsense phase-1. Host GPU 1 is selected at run time (compose / --gpus), not here.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        libgl1 \
        libatomic1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        llvm \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy scipy matplotlib pytest sionna-rt==2.2.0

COPY . .

CMD ["python", "scripts/run_pilot.py", "--root", "data", "--out", "results"]
