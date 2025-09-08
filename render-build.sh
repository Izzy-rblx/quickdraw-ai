#!/usr/bin/env bash
set -o errexit  # exit on error

# Install Git LFS and pull large files (like .keras model)
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt
