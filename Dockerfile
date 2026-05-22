FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    python3 \
    python3-pip \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY linux_manus_logger /app/linux_manus_logger

RUN cd /app/linux_manus_logger && make

ENV LD_LIBRARY_PATH=/app/linux_manus_logger/ManusSDK/lib
ENTRYPOINT ["/app/linux_manus_logger/manus_integrated_logger"]
