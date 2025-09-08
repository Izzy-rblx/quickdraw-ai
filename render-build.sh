#!/usr/bin/env bash
set -o errexit  # stop on error

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download model file from GitHub release if not already present
if [ ! -f quickdraw_model.keras ]; then
  echo "Downloading model from GitHub release..."
  curl -L -o quickdraw_model.keras \
    https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras
fi
