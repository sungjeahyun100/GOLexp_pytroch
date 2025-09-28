# 멀티스테이지 Dockerfile - GPU/CPU 자동 선택
ARG CUDA_VERSION=12.1
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${BASE_IMAGE} as gpu-stage

# GPU 빌드 스테이지
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential cmake git \
    gcc-12 g++-12 ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

WORKDIR /app
COPY . .

# Python 의존성 설치
RUN pip3 install torch torchvision numpy matplotlib tqdm argparse

# CMake 빌드
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 최종 스테이지
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 런타임 패키지만 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
RUN pip3 install torch torchvision numpy matplotlib tqdm argparse

# 빌드된 파일 복사
COPY --from=gpu-stage /app /app
WORKDIR /app

CMD ["bash"]