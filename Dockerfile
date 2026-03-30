# SmolVLA + MuJoCo. Ожидаем официальные образы pytorch/pytorch:* (torch + torchvision + torchaudio).
#
# Сборка по умолчанию:
#   docker build -t lerobot-workshop .
#
# Другой тег того же семейства:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime -t lerobot-workshop .
#   и синхронно поправьте docker/constraints.txt под версии из этого тега.
#
# Запуск с GPU и окном (Linux):
#   docker run --rm -it --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix lerobot-workshop

ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    unzip \
    libgl1 \
    libglu1-mesa \
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

WORKDIR /app

COPY requirements-docker.txt ./
COPY docker/constraints.txt docker/constraints.txt

RUN python -c "import torch; print('base torch:', torch.__version__)" \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-docker.txt -c docker/constraints.txt

RUN python -c "import mujoco, torch, torchvision, torchaudio, lerobot, transformers; print('ok:', torch.__version__)"

COPY . .

RUN if [ ! -d asset ]; then unzip -q asset.zip; fi

ENV PYTHONPATH=/app
