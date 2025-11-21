# CloudVina Phase 1 - AWS Setup Guide

This guide walks you through setting up AWS for CloudVina using the Free Tier.

## Step 1: Create AWS Account (if needed)

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the signup process (requires credit card, but won't be charged in Free Tier)

## Step 2: Setup AWS CLI

### Install AWS CLI

**Windows:**
```powershell
# Download and run the installer
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

**Mac/Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configure AWS CLI

```bash
aws configure
```

You'll need:
- **AWS Access Key ID**: Get from IAM Console → Users → Security Credentials
- **AWS Secret Access Key**: Provided when you create access key
- **Default region**: `us-east-1` (Free Tier eligible)
- **Default output format**: `json`

## Step 3: Create S3 Bucket

```bash
# Create bucket (must be globally unique name)
aws s3 mb s3://cloudvina-jobs-[your-username]

# Enable versioning (optional, for safety)
aws s3api put-bucket-versioning \
  --bucket cloudvina-jobs-[your-username] \
  --versioning-configuration Status=Enabled

# Set lifecycle policy to delete files after 30 days (save costs)
cat > lifecycle.json << 'EOF'
{
  "Rules": [{
    "Id": "DeleteOldJobs",
    "Status": "Enabled",
    "Prefix": "jobs/",
    "Expiration": { "Days": 30 }
  }]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket cloudvina-jobs-[your-username] \
  --lifecycle-configuration file://lifecycle.json
```

## Step 4: Create IAM Role for AWS Batch

This role allows Batch jobs to access S3.

```bash
# Create trust policy
cat > batch-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": [
        "ecs-tasks.amazonaws.com",
        "batch.amazonaws.com"
      ]
    },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
  --role-name CloudVinaJobRole \
  --assume-role-policy-document file://batch-trust-policy.json

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name CloudVinaJobRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Note the Role ARN - you'll need this later
aws iam get-role --role-name CloudVinaJobRole --query 'Role.Arn'
```

## Step 5: Create ECR Repository

```bash
# Create repo
aws ecr create-repository --repository-name cloudvina

# Get repo URI (save this)
aws ecr describe-repositories \
  --repository-names cloudvina \
  --query 'repositories[0].repositoryUri' \
  --output text
```

## Step 6: Setup AWS Batch (Manual via Console for now)

**Why manual?** AWS Batch setup via CLI is complex. For MVP, using the console is faster.

### 6a. Create Compute Environment

1. Go to [AWS Batch Console](https://console.aws.amazon.com/batch)
2. Click "Compute environments" → "Create"
3. Settings:
   - **Name**: `cloudvina-compute-free-tier`
   - **Provisioning model**: **Managed (recommended)**
   - **Instance type**: `t2.micro` (Free Tier)
   - **Min vCPUs**: `0` (pay only when running)
   - **Max vCPUs**: `2` (limit to avoid costs)
   - **Desired vCPUs**: `0`
   - **Spot or On-Demand**: **On-Demand** (for Free Tier stability)
4. Click "Create"

### 6b. Create Job Queue

1. Click "Job queues" → "Create"
2. Settings:
   - **Name**: `cloudvina-queue-free-tier`
   - **Priority**: `1`
   - **Compute environment**: Select `cloudvina-compute-free-tier`
3. Click "Create"

### 6c. Create Job Definition

1. Click "Job definitions" → "Create"
2. Settings:
   - **Name**: `cloudvina-job`
   - **Platform**: **Fargate** or **EC2** (choose EC2 for Free Tier)
   - **Execution role**: Select `CloudVinaJobRole` (created in Step 4)
   - **Image**: `[your-ecr-repo-uri]:latest` (from Step 5)
   - **Command**: Leave empty (uses ENTRYPOINT from Dockerfile)
   - **vCPUs**: `1`
   - **Memory**: `1024` MB
   - **Job role**: Select `CloudVinaJobRole`
3. **Environment variables** (add these):
   ```
   JOB_ID = (will be set per job)
   S3_BUCKET = cloudvina-jobs-[your-username]
   RECEPTOR_S3_KEY = (will be set per job)
   LIGAND_S3_KEY = (will be set per job)
   ```
4. Click "Create"

## Step 7: Test the Pipeline

### Push Docker Image to ECR

```bash
cd cloudvina/docker

# Build
docker build -t cloudvina:latest .

# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag cloudvina:latest [ecr-repo-uri]:latest

# Push
docker push [ecr-repo-uri]:latest
```

### Upload Test Files to S3

```bash
# Download sample files
mkdir test_files
cd test_files

# Download HIV protease receptor (example)
wget https://files.rcsb.org/download/1HVR.pdb -O receptor.pdb

# You'll need a ligand file - create a simple one or download
# For testing, you can use any small molecule in SDF format

# Upload to S3
aws s3 cp receptor.pdb s3://cloudvina-jobs-[your-username]/test/receptor.pdb
aws s3 cp ligand.sdf s3://cloudvina-jobs-[your-username]/test/ligand.sdf
```

### Submit Test Job

```bash
aws batch submit-job \
  --job-name test-docking-001 \
  --job-queue cloudvina-queue-free-tier \
  --job-definition cloudvina-job \
  --container-overrides '{
    "environment": [
      {"name": "JOB_ID", "value": "test-001"},
      {"name": "RECEPTOR_S3_KEY", "value": "test/receptor.pdb"},
      {"name": "LIGAND_S3_KEY", "value": "test/ligand.sdf"}
    ]
  }'
```

### Monitor Job

```bash
# Get job ID from submit-job output, then:
aws batch describe-jobs --jobs [job-id]

# Check S3 for results
aws s3 ls s3://cloudvina-jobs-[your-username]/jobs/test-001/
```

## Cost Monitoring

### Set Billing Alerts

1. Go to [Billing Dashboard](https://console.aws.amazon.com/billing)
2. Click "Budgets" → "Create budget"
3. Set alerts at $10, $50, $100

### Track Free Tier Usage

1. Go to Billing Dashboard
2. Click "Free Tier"
3. Monitor:
   - EC2 hours used (750/month limit)
   - S3 storage (5GB limit)

## Troubleshooting

**Issue: Job stuck in "RUNNABLE"**
- Check compute environment is "ENABLED"
- Verify max vCPUs > 0
- Check IAM role permissions

**Issue: "AccessDenied" errors**
- Verify job role has S3 permissions
- Check bucket policy allows the role

**Issue: Docker image pull failed**
- Ensure image was pushed to ECR
- Verify job definition has correct image URI
- Check execution role has ECR permissions

## Next Steps

Once this works:
1. ✅ Test end-to-end docking
2. Build FastAPI backend (Phase 2)
3. Switch to Spot Instances for production

## Quick Reference

**Your Resources:**
- S3 Bucket: `s3://cloudvina-jobs-[your-username]`
- ECR Repo: `[account-id].dkr.ecr.us-east-1.amazonaws.com/cloudvina`
- Job Queue: `cloudvina-queue-free-tier`
- Compute Env: `cloudvina-compute-free-tier`

**Useful Commands:**
```bash
# List jobs
aws batch list-jobs --job-queue cloudvina-queue-free-tier

# Describe job
aws batch describe-jobs --jobs [job-id]

# Cancel job
aws batch cancel-job --job-id [job-id] --reason "Testing"

# Check S3 results
aws s3 ls s3://cloudvina-jobs-[your-username]/jobs/ --recursive
```
