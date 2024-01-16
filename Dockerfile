# Base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set up workspace
WORKDIR /workspace
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
