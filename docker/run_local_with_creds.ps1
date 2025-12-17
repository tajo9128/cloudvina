# CloudVina Docker Runner Helper
# Usage: ./run_local_with_creds.ps1
# Prompts for AWS Keys and runs the container

$ErrorActionPreference = "Stop"

Write-Host "--- CloudVina Local Runner ---" -ForegroundColor Cyan

# 1. Ask for Credentials (if not set in env)
$AwsAccessKey = $env:AWS_ACCESS_KEY_ID
if (-not $AwsAccessKey) {
    $AwsAccessKey = Read-Host "Enter AWS Access Key ID"
}

$AwsSecretKey = $env:AWS_SECRET_ACCESS_KEY
if (-not $AwsSecretKey) {
    $AwsSecretKey = Read-Host "Enter AWS Secret Access Key" -AsSecureString
    # Convert SecureString back to plain text for Docker env var (runs locally so acceptable for dev)
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($AwsSecretKey)
    $AwsSecretKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
}

$AwsRegion = $env:AWS_DEFAULT_REGION
if (-not $AwsRegion) {
    $AwsRegion = Read-Host "Enter AWS Region (default: us-east-1)"
    if (-not $AwsRegion) { $AwsRegion = "us-east-1" }
}

# 2. Ask for Job Files
$JobId = Read-Host "Enter Job ID (e.g. test-001)"
if (-not $JobId) { $JobId = "test-001" }

$Receptor = Read-Host "Enter Receptor S3 Key (e.g. receptor.pdb)"
if (-not $Receptor) { $Receptor = "receptor.pdb" }

$Ligand = Read-Host "Enter Ligand S3 Key (e.g. ligand.sdf)"
if (-not $Ligand) { $Ligand = "ligand.sdf" }

# 3. Build Docker Image (Optional but good check)
Write-Host "`nChecking/Building Docker Image..." -ForegroundColor Yellow
docker build -t cloudvina:latest ..

# 4. Run Container
Write-Host "`nStarting Container..." -ForegroundColor Green

# Mount usage:
# We assume the user wants to mount the current test_data folder to /app/work 
# or just let the container download from S3 if keys are provided.
# If S3 keys are provided, we don't strictly need a volume unless we want to see output locally.
# We'll mount the ../test_data folder to /app/work so outputs are saved there.

$WorkDir = Join-Path (Get-Location).Path "..\test_data"
if (-not (Test-Path $WorkDir)) {
    New-Item -ItemType Directory -Path $WorkDir | Out-Null
}

Write-Host "Mounting: $WorkDir"

docker run --rm `
    -v "${WorkDir}:/app/work" `
    -e AWS_ACCESS_KEY_ID="$AwsAccessKey" `
    -e AWS_SECRET_ACCESS_KEY="$AwsSecretKey" `
    -e AWS_DEFAULT_REGION="$AwsRegion" `
    -e JOB_ID="$JobId" `
    -e RECEPTOR_S3_KEY="$Receptor" `
    -e LIGAND_S3_KEY="$Ligand" `
    -e S3_BUCKET="cloudvina-jobs" `
    cloudvina:latest

Write-Host "`nDone." -ForegroundColor Cyan
