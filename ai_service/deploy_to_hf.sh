#!/bin/bash
# deploy_to_hf.sh - Deploy ai_service to HuggingFace Spaces

echo "🚀 BioDockify AI Backend Deployment Script"
echo "=========================================="

# Configuration - CHANGE THESE
HF_USERNAME="tajo9128"
SPACE_NAME="biodockify-ai"
HF_REMOTE="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

# Navigate to ai_service
cd ai_service || { echo "❌ Error: ai_service directory not found"; exit 1; }

echo "📦 Initializing Git repository for ai_service..."

# Initialize fresh git repo for this deployment
rm -rf .git
git init
git add .
git commit -m "Deploy: $(date +%Y-%m-%d)"

echo "🔗 Adding HuggingFace remote..."
git remote add hf "${HF_REMOTE}"

echo "🚢 Pushing to HuggingFace Spaces..."
echo "   (You will be prompted for username and token)"
echo "   Username: ${HF_USERNAME}"
echo "   Password: Get token from https://huggingface.co/settings/tokens"
echo ""

git push --force hf master:main

echo ""
echo "✅ Deployment complete!"
echo "🌐 API will be live at: https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo "📚 API Docs at: https://${HF_USERNAME}-${SPACE_NAME}.hf.space/docs"
