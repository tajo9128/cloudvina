#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Build Start: Installing dependencies..."
pip install --upgrade pip

# Pre-install build dependencies (including sklearn for ODDT setup)
pip install six numpy setuptools wheel scikit-learn joblib pandas scipy

# Install ODDT with --no-build-isolation to use the pre-installed 'six'
echo "Installing ODDT..."
pip install --no-build-isolation oddt>=0.7

# Install remaining requirements
echo "Installing requirements.txt..."
pip install -r api/requirements.txt

echo "Build Complete!"
