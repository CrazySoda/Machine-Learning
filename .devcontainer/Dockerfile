# Use the official PyTorch image with CUDA support which matches requirements.txt
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv (The Python Package Manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set up a working directory
WORKDIR /workspace

# Copy requirements.txt from the host
COPY requirements.txt .

# Install Python dependencies using uv
# --system installs into the system python environment, which is fine for a container
RUN uv pip install --system --no-cache -r requirements.txt

# Create a non-root user matching the default vscode user (1000)
# (Optional, as the devcontainer 'remoteUser' setting or features can handle this)
