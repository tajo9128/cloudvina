#!/bin/bash
# BioDockify Quick Debug Commands
# Run this script to quickly check common issues

set -e

echo "========================================="
echo "BioDockify Quick Debug Commands"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# 1. Check Environment Variables
echo "1. Checking Environment Variables..."
echo "-----------------------------------"

if [ -z "$AWS_REGION" ]; then
    print_warning "AWS_REGION not set (default: us-east-1)"
else
    print_success "AWS_REGION: $AWS_REGION"
fi

if [ -z "$S3_BUCKET" ]; then
    print_warning "S3_BUCKET not set (default: BioDockify-jobs-use1-1763775915)"
else
    print_success "S3_BUCKET: $S3_BUCKET"
fi

if [ -z "$BATCH_JOB_QUEUE" ]; then
    print_warning "BATCH_JOB_QUEUE not set (default: cloudvina-fargate-queue)"
else
    print_success "BATCH_JOB_QUEUE: $BATCH_JOB_QUEUE"
fi

if [ -z "$BATCH_JOB_DEFINITION" ]; then
    print_warning "BATCH_JOB_DEFINITION not set (default: cloudvina-job)"
else
    print_success "BATCH_JOB_DEFINITION: $BATCH_JOB_DEFINITION"
fi

echo ""

# 2. Check AWS Credentials
echo "2. Checking AWS Credentials..."
echo "-----------------------------------"

if aws sts get-caller-identity &> /dev/null; then
    IDENTITY=$(aws sts get-caller-identity --query Account --output text)
    print_success "AWS credentials found"
    print_info "Account: $IDENTITY"
else
    print_error "AWS credentials not found or invalid"
    echo "   Run: aws configure"
fi

echo ""

# 3. Check S3 Bucket
echo "3. Checking S3 Bucket Access..."
echo "-----------------------------------"

BUCKET="${S3_BUCKET:-BioDockify-jobs-use1-1763775915}"

if aws s3 ls "s3://$BUCKET" &> /dev/null; then
    print_success "Bucket '$BUCKET' is accessible"
    
    # List recent job folders
    echo ""
    print_info "Recent job folders:"
    aws s3 ls "s3://$BUCKET/jobs/" --recursive | grep "/$" | head -5 | while read line; do
        echo "   $line"
    done
else
    print_error "Cannot access bucket '$BUCKET'"
fi

echo ""

# 4. Check AWS Batch Queue
echo "4. Checking AWS Batch Queue..."
echo "-----------------------------------"

QUEUE="${BATCH_JOB_QUEUE:-cloudvina-fargate-queue}"

if aws batch describe-job-queues --job-queues "$QUEUE" &> /dev/null; then
    print_success "Job queue '$QUEUE' exists"
    
    STATUS=$(aws batch describe-job-queues --job-queues "$QUEUE" --query 'jobQueues[0].status' --output text)
    STATE=$(aws batch describe-job-queues --job-queues "$QUEUE" --query 'jobQueues[0].state' --output text)
    
    print_info "Status: $STATUS, State: $STATE"
else
    print_error "Job queue '$QUEUE' not found"
    echo "   Run: aws batch describe-job-queues"
fi

echo ""

# 5. Check Job Definition
echo "5. Checking Job Definition..."
echo "-----------------------------------"

JOB_DEF="${BATCH_JOB_DEFINITION:-cloudvina-job}"

# Handle malformed definition
if [[ "$JOB_DEF" == :* ]]; then
    print_warning "Job definition starts with ':' (malformed)"
    JOB_DEF="biodockify-all-fixes$JOB_DEF"
    print_info "Trying corrected: $JOB_DEF"
fi

if aws batch describe-job-definitions --job-definitions "$JOB_DEF" &> /dev/null; then
    print_success "Job definition '$JOB_DEF' exists"
    
    STATUS=$(aws batch describe-job-definitions --job-definitions "$JOB_DEF" --query 'jobDefinitions[0].status' --output text)
    TYPE=$(aws batch describe-job-definitions --job-definitions "$JOB_DEF" --query 'jobDefinitions[0].type' --output text)
    
    print_info "Status: $STATUS, Type: $TYPE"
else
    print_error "Job definition '$JOB_DEF' not found"
    echo "   Run: aws batch describe-job-definitions --status ACTIVE"
fi

echo ""

# 6. Check Recent Batch Jobs
echo "6. Checking Recent Batch Jobs..."
echo "-----------------------------------"

if aws batch list-jobs --job-queue "$QUEUE" --max-results 5 &> /dev/null; then
    print_success "Recent jobs found"
    
    echo ""
    print_info "Last 5 jobs:"
    aws batch list-jobs --job-queue "$QUEUE" --max-results 5 --output table
else
    print_warning "No recent jobs or queue not accessible"
fi

echo ""

# 7. Summary
echo "========================================="
echo "Summary"
echo "========================================="
echo ""
echo "If all checks passed, you can submit a test job."
echo ""
echo "Common issues:"
echo "  - AWS credentials: Run 'aws configure'"
echo "  - S3 bucket: Check bucket name and permissions"
echo "  - Batch queue: Verify queue exists and is ACTIVE"
echo "  - Job definition: Check definition name and status"
echo ""
echo "For detailed debugging, run:"
echo "  python debug_docking_jobs.py"
echo ""
echo "For troubleshooting guide, see:"
echo "  DEBUGGING_GUIDE.md"
echo ""
