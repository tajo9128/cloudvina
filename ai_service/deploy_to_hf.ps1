# deploy_to_hf.ps1 - Deploy ai_service to HuggingFace Spaces (Windows)

Write-Host "🚀 BioDockify AI Backend Deployment Script" -ForegroundColor Green
Write-Host "==========================================`n"

# Configuration - CHANGE THESE
$HF_USERNAME = "tajo9128"
$SPACE_NAME = "biodockify-ai"
$HF_REMOTE = "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Navigate to ai_service
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "📦 Initializing Git repository for ai_service..." -ForegroundColor Cyan

# Remove old git if exists
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
}

# Initialize fresh git repo
git init
git add .
git commit -m "Deploy: $(Get-Date -Format 'yyyy-MM-dd')"

Write-Host "`n🔗 Adding HuggingFace remote..." -ForegroundColor Cyan
git remote add hf $HF_REMOTE

Write-Host "`n🚢 Pushing to HuggingFace Spaces..." -ForegroundColor Yellow
Write-Host "   (You will be prompted for username and token)"
Write-Host "   Username: $HF_USERNAME"
Write-Host "   Password: Get token from https://huggingface.co/settings/tokens`n"

git push --force hf master:main

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "🌐 API will be live at: https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Magenta
Write-Host "📚 API Docs at: https://$HF_USERNAME-$SPACE_NAME.hf.space/docs" -ForegroundColor Magenta
