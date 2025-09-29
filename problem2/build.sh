#!/bin/bash
echo "Building autoencoder training container..."
docker build -t arxiv-embeddings:latest .
echo "Build complete"