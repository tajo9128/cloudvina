#!/bin/bash
# AWS ECR Deployment Script for CloudShell
# Usage: ./AWS_Batch_Deploy.sh

set -e # Exit on error

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="cloudvina-fargate-repo"  # Adjust if your repo name differs
IMAGE_TAG="latest"
JOB_DEF_NAME="biodockify-all-fixes"

echo "🚀 Deploying to AWS Batch (ECR)..."
echo "   Region: $REGION"
echo "   Account: $ACCOUNT_ID"
echo "   Repo: $ECR_REPO"

# 1. Login to ECR
echo "🔑 Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 2. Build Docker Image
echo "🔨 Building Image..."
# We assume we are in the 'api' folder or root? 
# Script is placed in api/, but we need to build from PROJECT ROOT
# Resolve absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to Parent Directory (Project Root)
cd "$SCRIPT_DIR/.." || exit

echo "📂 Context: $(pwd)"
docker build -t $ECR_REPO .

# 3. Tag Image
echo "🏷️ Tagging Image..."
docker tag $ECR_REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# 4. Push Image
echo "⬆️ Pushing to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

echo "✅ Image Update Complete!"
echo "   The AWS Batch Job Definition '$JOB_DEF_NAME' will now use this new image for all future jobs (if set to use :latest)."
echo "   You may need to register a new job definition revision if :latest is not cached."
