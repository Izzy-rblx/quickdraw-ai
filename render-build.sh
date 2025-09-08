#!/usr/bin/env bash
set -o errexit  # exit on any error

# Upgrade pip
pip install --upgrade pip

MODEL_FILE="quickdraw_model.keras"
MODEL_URL="https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras"
MODEL_SHA256="af274f007abc6d93ee760177affbc37b3b1674cefd811389cae661017dcd6784"

# Download model if missing
if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading $MODEL_FILE..."
  curl -L -o "$MODEL_FILE" "$MODEL_URL"

  echo "Verifying checksum..."
  echo "$MODEL_SHA256  $MODEL_FILE" | sha256sum -c -

  echo "Model verified successfully."
fi

# Install dependencies
pip install -r requirements.txt
