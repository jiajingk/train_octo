# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3 and pip3 to use Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Create a working directory
WORKDIR /app

# Clone the repository and install it in editable mode
RUN git clone https://github.com/octo-models/octo.git . \
    && pip3 install -e .

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy the hello_jax.py script into the container
COPY minimal_inference.py .

# Set the entrypoint to run the script
ENTRYPOINT ["python3", "minimal_inference.py"]
