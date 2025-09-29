#!/bin/bash

# Check for port argument
PORT=${1:-8080}

# Validate port is numeric
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port must be numeric"
    exit 1
fi

# Check port range
if [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: Port must be between 1024 and 65535"
    exit 1
fi

echo "Starting ArXiv API server on port $PORT"
echo "Access at: http://localhost:$PORT"
echo ""
echo "Available endpoints:"
echo "  GET /papers"
echo "  GET /papers/{arxiv_id}"
echo "  GET /search?q={query}"
echo "  GET /stats"
echo ""

# Run container
docker run --rm \
    --name arxiv-server \
    -p "$PORT:8080" \
    arxiv-server:latest