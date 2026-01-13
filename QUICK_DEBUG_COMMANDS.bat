@echo off
REM BioDockify Quick Debug Commands (Windows)
REM Run this script to quickly check common issues

echo =========================================
echo BioDockify Quick Debug Commands
echo =========================================
echo.

REM 1. Check Environment Variables
echo 1. Checking Environment Variables...
echo -----------------------------------

if "%AWS_REGION%"=="" (
    echo [WARNING] AWS_REGION not set ^(default: us-east-1^)
) else (
    echo [OK] AWS_REGION: %AWS_REGION%
)

if "%S3_BUCKET%"=="" (
    echo [WARNING] S3_BUCKET not set ^(default: BioDockify-jobs-use1-1763775915^)
) else (
    echo [OK] S3_BUCKET: %S3_BUCKET%
)

if "%BATCH_JOB_QUEUE%"=="" (
    echo [WARNING] BATCH_JOB_QUEUE not set ^(default: cloudvina-fargate-queue^)
) else (
    echo [OK] BATCH_JOB_QUEUE: %BATCH_JOB_QUEUE%
)

if "%BATCH_JOB_DEFINITION%"=="" (
    echo [WARNING] BATCH_JOB_DEFINITION not set ^(default: cloudvina-job^)
) else (
    echo [OK] BATCH_JOB_DEFINITION: %BATCH_JOB_DEFINITION%
)

echo.

REM 2. Check AWS Credentials
echo 2. Checking AWS Credentials...
echo -----------------------------------

aws sts get-caller-identity >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] AWS credentials found
    for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set ACCOUNT=%%i
    echo [INFO] Account: %ACCOUNT%
) else (
    echo [ERROR] AWS credentials not found or invalid
    echo    Run: aws configure
)

echo.

REM 3. Check S3 Bucket
echo 3. Checking S3 Bucket Access...
echo -----------------------------------

if "%S3_BUCKET%"=="" (
    set BUCKET=BioDockify-jobs-use1-1763775915
) else (
    set BUCKET=%S3_BUCKET%
)

aws s3 ls "s3://%BUCKET%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Bucket '%BUCKET%' is accessible
    
    echo.
    echo [INFO] Recent job folders:
    aws s3 ls "s3://%BUCKET%/jobs/" --recursive | findstr "/$" | more +0
) else (
    echo [ERROR] Cannot access bucket '%BUCKET%'
)

echo.

REM 4. Check AWS Batch Queue
echo 4. Checking AWS Batch Queue...
echo -----------------------------------

if "%BATCH_JOB_QUEUE%"=="" (
    set QUEUE=cloudvina-fargate-queue
) else (
    set QUEUE=%BATCH_JOB_QUEUE%
)

aws batch describe-job-queues --job-queues "%QUEUE%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Job queue '%QUEUE%' exists
    
    for /f "tokens=*" %%i in ('aws batch describe-job-queues --job-queues "%QUEUE%" --query "jobQueues[0].status" --output text') do set STATUS=%%i
    for /f "tokens=*" %%i in ('aws batch describe-job-queues --job-queues "%QUEUE%" --query "jobQueues[0].state" --output text') do set STATE=%%i
    
    echo [INFO] Status: %STATUS%, State: %STATE%
) else (
    echo [ERROR] Job queue '%QUEUE%' not found
    echo    Run: aws batch describe-job-queues
)

echo.

REM 5. Check Job Definition
echo 5. Checking Job Definition...
echo -----------------------------------

if "%BATCH_JOB_DEFINITION%"=="" (
    set JOB_DEF=cloudvina-job
) else (
    set JOB_DEF=%BATCH_JOB_DEFINITION%
)

REM Handle malformed definition
echo %JOB_DEF% | findstr /B ":" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Job definition starts with ':' ^(malformed^)
    set JOB_DEF=biodockify-all-fixes%JOB_DEF%
    echo [INFO] Trying corrected: %JOB_DEF%
)

aws batch describe-job-definitions --job-definitions "%JOB_DEF%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Job definition '%JOB_DEF%' exists
    
    for /f "tokens=*" %%i in ('aws batch describe-job-definitions --job-definitions "%JOB_DEF%" --query "jobDefinitions[0].status" --output text') do set STATUS=%%i
    for /f "tokens=*" %%i in ('aws batch describe-job-definitions --job-definitions "%JOB_DEF%" --query "jobDefinitions[0].type" --output text') do set TYPE=%%i
    
    echo [INFO] Status: %STATUS%, Type: %TYPE%
) else (
    echo [ERROR] Job definition '%JOB_DEF%' not found
    echo    Run: aws batch describe-job-definitions --status ACTIVE
)

echo.

REM 6. Check Recent Batch Jobs
echo 6. Checking Recent Batch Jobs...
echo -----------------------------------

aws batch list-jobs --job-queue "%QUEUE%" --max-results 5 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Recent jobs found
    
    echo.
    echo [INFO] Last 5 jobs:
    aws batch list-jobs --job-queue "%QUEUE%" --max-results 5 --output table
) else (
    echo [WARNING] No recent jobs or queue not accessible
)

echo.

REM 7. Summary
echo =========================================
echo Summary
echo =========================================
echo.
echo If all checks passed, you can submit a test job.
echo.
echo Common issues:
echo   - AWS credentials: Run 'aws configure'
echo   - S3 bucket: Check bucket name and permissions
echo   - Batch queue: Verify queue exists and is ACTIVE
echo   - Job definition: Check definition name and status
echo.
echo For detailed debugging, run:
echo   python debug_docking_jobs.py
echo.
echo For troubleshooting guide, see:
echo   DEBUGGING_GUIDE.md
echo.

pause
