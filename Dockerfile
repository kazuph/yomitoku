FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt -y update && apt -y upgrade

ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    poppler-utils \
    libopencv-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} --version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /workspace

COPY . .

RUN uv sync --frozen --no-cache
RUN uv pip install fastapi uvicorn python-multipart
RUN uv pip install -e .

CMD ["python", "app.py"]
