#!/bin/bash

source .env

echo "Building Docker image..."
docker build -t langchain-exp .

echo "Starting container in background..."
docker run -d \
  --name langchain-container \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd):/app \
  --entrypoint tail \
  langchain-exp -f /dev/null

echo "Container is ready. Use ./start.sh <filename.py> to run scripts."
