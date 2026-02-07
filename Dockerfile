# DPVO - Deep Patch Visual Odometry/SLAM
# Docker image for development and inference

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    ca-certificates \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libegl1-mesa-dev \
    libxkbcommon-dev \
    libwayland-dev \
    wayland-protocols \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libpng-dev \
    libjpeg-dev \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set default user
USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# Default command
CMD ["/bin/bash"]
