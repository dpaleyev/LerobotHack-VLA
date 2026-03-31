# SmolVLA + MuJoCo training/inference container.
# Python 3.12 required by lerobot>=0.5.0; official pytorch images ship 3.11,
# so we install 3.12 from deadsnakes and reinstall torch from pip.
#
# Сборка:
#   docker build -t lerobot-workshop .
#
# Запуск с GPU и окном (Linux):
#   docker run --rm -it --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix lerobot-workshop

ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3.12-tk \
    build-essential \
    git \
    unzip \
    libgl1 \
    libglu1-mesa \
    libosmesa6 \
    libglfw3 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libxkbcommon0 \
    libgomp1 \
    libjpeg8 \
    libpng16-16 \
    libtiff5 \
    libusb-1.0-0 \
    scrot \
    && rm -rf /var/lib/apt/lists/*

# Create a clean Python 3.12 venv (sidesteps conda 3.11 entirely)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && python --version && pip --version

WORKDIR /app

COPY requirements-docker.txt ./

RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements-docker.txt

RUN python -c "import sys; print('python:', sys.version)" \
    && python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" \
    && python -c "import lerobot; print('lerobot:', lerobot.__version__)" \
    && python -c "import mujoco; print('mujoco:', mujoco.__version__)"

COPY . .

RUN if [ ! -d asset ]; then unzip -q asset.zip; fi

ENV PYTHONPATH=/app
