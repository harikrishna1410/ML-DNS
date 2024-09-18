# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install MPI and other necessary build tools
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the ml_dns package
RUN pip install torch numpy mpi4py h5py matplotlib

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME TEST-ML-DNS
