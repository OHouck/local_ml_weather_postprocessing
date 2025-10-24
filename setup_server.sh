#!/bin/bash
set -e
echo "Setting up for GPU cluster (CUDA)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv sync --extra cuda
echo "Setup complete!"
