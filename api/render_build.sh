#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Build Start: Installing dependencies..."
python -m pip install --upgrade pip

# Pre-install build dependencies (including sklearn for ODDT setup)
python -m pip install six numpy setuptools wheel scikit-learn joblib pandas scipy

# Install ODDT with --no-build-isolation to use the pre-installed 'six'
echo "Installing ODDT..."
python -m pip install --no-build-isolation oddt>=0.7

# Install remaining requirements
echo "Installing requirements.txt..."
python -m pip install -r api/requirements.txt

echo "Build Complete!"
