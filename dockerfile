FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3.9-tk \
    gcc \
    build-essential \
    pkg-config \
    libcairo2-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN python3.9 -m venv venv
RUN ./venv/bin/pip install --upgrade pip
RUN ./venv/bin/pip install -r requirements.txt

CMD ["./venv/bin/python", "-m", "main"]