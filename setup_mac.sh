#!/bin/bash
set -e
echo "Setting up for Mac (Apple Silicon)..."
uv sync --no-install-project
uv pip install torch torchvision torchaudio
echo "Setup complete!"
