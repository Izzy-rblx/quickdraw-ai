#!/usr/bin/env bash
set -o errexit  # exit on error

# Install git-lfs (needed to pull your model file)
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt
