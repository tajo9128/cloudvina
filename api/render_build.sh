#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Build Start: Installing dependencies..."
python -m pip install --upgrade pip

# Pre-install build dependencies
python -m pip install six numpy setuptools wheel scikit-learn joblib pandas scipy

# ODDT removed - not needed for API, only Docker container

# Install remaining requirements
echo "Installing requirements.txt..."
python -m pip install -r api/requirements.txt

echo "Build Complete!"
