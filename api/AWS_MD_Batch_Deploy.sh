#!/bin/bash
# AWS ECR Deployment Script for MD Simulation Engine
# Usage: ./AWS_MD_Batch_Deploy.sh

set -e # Exit on error

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="cloudvina"  # Use shared repo
IMAGE_TAG="md-simulation" # Specific tag for MD engine
JOB_DEF_NAME="md-simulation-job-def"

echo "🚀 Deploying MD Simulation Engine to AWS Batch..."
echo "   Region: $REGION"
echo "   Account: $ACCOUNT_ID"
echo "   Repo: $ECR_REPO"

# 1. Login to ECR
echo "🔑 Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 2. Build Docker Image
echo "🔨 Building MD Image..."
# Navigate to Project Root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit
echo "📂 Context: $(pwd)"

# Check for Model File
if [ ! -f "md_stability_model.pkl" ]; then
    echo "❌ ERROR: md_stability_model.pkl not found in root!"
    echo "   Please download the model bundle before deploying."
    exit 1
fi

docker build -f docker/Dockerfile.md_simulation -t $ECR_REPO .

# 3. Tag Image
echo "🏷️ Tagging Image..."
docker tag $ECR_REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# 4. Push Image
echo "⬆️ Pushing to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

echo "✅ MD Engine Update Complete!"
echo "   Job Definition: $JOB_DEF_NAME"
