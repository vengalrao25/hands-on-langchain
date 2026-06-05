#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./start.sh <filename.py>"
  echo "Example: ./start.sh main.py"
  exit 1
fi

source .env

echo "Running $1..."
docker exec langchain-container python $1
