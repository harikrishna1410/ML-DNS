# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install MPI and other necessary build tools
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the ml_dns package
RUN pip install --no-cache-dir -e .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME ML-DNS

# Create a directory for examples
RUN mkdir /examples

# Copy examples to the /examples directory and start a bash session
CMD cp -r /app/example/* /examples/ && cd /examples && /bin/bash