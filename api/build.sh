#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Building CloudVina API..."

# Upgrade pip
pip install --upgrade pip

# Install six first (required for oddt setup.py)
pip install six

# Install dependencies with no build isolation to ensure six is visible to oddt
pip install --no-build-isolation -r requirements.txt
pip install itsdangerous

echo "Build completed successfully!"
