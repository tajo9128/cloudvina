# Quick Test Script for Format Conversion
# Builds Docker image and runs tests

Write-Host "Building test Docker image..." -ForegroundColor Cyan
docker build -f Dockerfile.test -t biodockify-format-test .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning format conversion tests..." -ForegroundColor Cyan
docker run --rm biodockify-format-test

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Some tests failed. Check output above." -ForegroundColor Red
}
