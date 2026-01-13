# BioDockify Docking Job Debugging Guide

This guide helps you diagnose and fix issues with docking jobs in the BioDockify project.

## Quick Start

Run the automated debugging script:

```bash
cd cloudvina
python debug_docking_jobs.py
```

This will check:
- ✅ Environment configuration
- ✅ AWS credentials and connectivity
- ✅ S3 bucket access
- ✅ AWS Batch configuration
- ✅ Recent job status
- ✅ Common configuration issues

## Common Issues and Solutions

### 1. AWS Credentials Not Found

**Symptoms:**
```
✗ No AWS credentials found
```

**Solution:**
```bash
# Option 1: Configure AWS CLI
aws configure

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. S3 Bucket Access Denied

**Symptoms:**
```
✗ Access denied to bucket 'BioDockify-jobs-use1-1763775915'
```

**Solutions:**

**A. Check bucket exists:**
```bash
aws s3 ls s3://BioDockify-jobs-use1-1763775915
```

**B. Verify bucket region:**
```bash
aws s3api get-bucket-location --bucket BioDockify-jobs-use1-1763775915
```

**C. Check IAM permissions:**
Ensure your IAM user has these permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:*"
      ],
      "Resource": [
        "arn:aws:s3:::BioDockify-jobs-use1-1763775915",
        "arn:aws:s3:::BioDockify-jobs-use1-1763775915/*"
      ]
    }
  ]
}
```

### 3. AWS Batch Job Queue Not Found

**Symptoms:**
```
✗ Job queue 'cloudvina-fargate-queue' not found
```

**Solution:**

**A. List available job queues:**
```bash
aws batch describe-job-queues
```

**B. Update environment variable:**
```bash
export BATCH_JOB_QUEUE=your_actual_queue_name
```

**C. Create job queue (if needed):**
Follow the setup in [`AWS_SETUP.md`](AWS_SETUP.md) or [`AWS_Batch_Deploy.sh`](AWS_Batch_Deploy.sh)

### 4. Job Definition Not Found

**Symptoms:**
```
✗ Job definition 'cloudvina-job' not found
```

**Solution:**

**A. List available job definitions:**
```bash
aws batch describe-job-definitions --status ACTIVE
```

**B. Update environment variable:**
```bash
export BATCH_JOB_DEFINITION=your_actual_job_definition
```

**C. Check if definition is malformed:**
```bash
# If it starts with ':', it's malformed
echo $BATCH_JOB_DEFINITION
# Should be like: biodockify-all-fixes:latest
# Not: :latest
```

### 5. Jobs Stuck in SUBMITTED/RUNNABLE State

**Symptoms:**
Jobs remain in SUBMITTED or RUNNABLE state for long periods.

**Solutions:**

**A. Check compute environment status:**
```bash
aws batch describe-compute-environments
```

**B. Check if compute environment is ACTIVE:**
```bash
aws batch describe-job-queues --job-queues cloudvina-fargate-queue
```

**C. Check CloudWatch logs for compute environment issues**

**D. Verify vCPU limits:**
```bash
aws service-quotas list-service-quotas --service-code batch
```

### 6. Jobs Failing with "Receptor conversion failed"

**Symptoms:**
```
✗ Receptor conversion failed: PDBQT file too short
```

**Solutions:**

**A. Check receptor file format:**
```bash
# Download the receptor file
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/receptor_input.pdb receptor.pdb

# Check if it's valid
head -20 receptor.pdb
```

**B. Verify receptor has ATOM records:**
```bash
grep "^ATOM\|^HETATM" receptor.pdb | wc -l
# Should have > 100 atoms
```

**C. Check for corruption:**
```bash
file receptor.pdb
# Should be "ASCII text" or similar
```

**D. Try converting manually:**
```bash
obabel receptor.pdb -O receptor.pdbqt -xr
```

### 7. Jobs Failing with "Ligand conversion failed"

**Symptoms:**
```
✗ Ligand conversion failed: SMILES parsing error
```

**Solutions:**

**A. Check ligand file format:**
```bash
# Download the ligand file
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/ligand_input.sdf ligand.sdf

# Check if it's valid
head -20 ligand.sdf
```

**B. Verify ligand has atoms:**
```bash
# For SDF
grep "^ " ligand.sdf | head -5

# For SMILES
cat ligand.smi
```

**C. Try converting manually:**
```bash
# For SMILES to PDBQT
obabel -:"CCO" -O ligand.pdbqt -xr

# For SDF to PDBQT
obabel ligand.sdf -O ligand.pdbqt -xr
```

### 8. Autoboxing Issues

**Symptoms:**
```
⚠️ No co-crystallized ligand detected. Using protein center
```

**Solutions:**

**A. Check if receptor has HETATM ligand:**
```bash
# Download receptor
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/receptor_input.pdb receptor.pdb

# Check for HETATM
grep "^HETATM" receptor.pdb | head -20
```

**B. Manually specify grid parameters:**
```python
# When starting job, provide explicit grid params
grid_params = {
    'grid_center_x': 10.5,
    'grid_center_y': 20.3,
    'grid_center_z': 15.7,
    'grid_size_x': 22.0,
    'grid_size_y': 22.0,
    'grid_size_z': 22.0
}
```

**C. Use a different receptor with known binding site**

### 9. Consensus Engine Issues

**Symptoms:**
```
✗ Consensus: Vina failed
✗ Consensus: RF failed
✗ Consensus: Gnina failed
```

**Solutions:**

**A. Try with simple Vina engine:**
```python
# In job start request
engine = 'vina'  # Instead of 'consensus'
```

**B. Check Vina binary exists:**
```bash
# In Docker container
which vina
/usr/local/bin/vina --version
```

**C. Check Gnina binary exists:**
```bash
which gnina
/usr/local/bin/gnina --version
```

**D. Check RF model service:**
```python
# In docker/main.py or similar
from services.rf_model_service import RFModelService
# Should import without errors
```

### 10. Timeout Issues

**Symptoms:**
```
✗ Gnina timed out after 300 seconds
```

**Solutions:**

**A. Increase timeout in [`docking_engine.py`](api/services/docking_engine.py):**
```python
# Line 364
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min
```

**B. Reduce exhaustiveness:**
```python
grid_params = {
    'exhaustiveness': 4  # Instead of 16
}
```

**C. Use smaller search box:**
```python
grid_params = {
    'size_x': 18.0,
    'size_y': 18.0,
    'size_z': 18.0
}
```

## Manual Debugging Steps

### Step 1: Check Job Status

```bash
# Using the API
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-api.com/jobs/{job_id}/status

# Or check database directly (if you have access)
```

### Step 2: Check AWS Batch Job

```bash
# Get batch job ID from database or API response
BATCH_JOB_ID="your_batch_job_id"

# Check status
aws batch describe-jobs --jobs $BATCH_JOB_ID

# Check CloudWatch logs
aws logs tail /aws/batch/job --follow
```

### Step 3: Check S3 Files

```bash
# List job files
aws s3 ls s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/

# Download and inspect files
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/log.txt .
cat log.txt

aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/config.txt .
cat config.txt
```

### Step 4: Check Container Logs

```bash
# Find the task ARN from batch job
aws batch describe-jobs --jobs $BATCH_JOB_ID | jq -r '.jobs[0].container.taskArn'

# Get CloudWatch log stream
aws logs tail /aws/batch/job --follow
```

### Step 5: Test Locally

```bash
# Download job files
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/receptor.pdbqt .
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/ligand.pdbqt .
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/config.txt .

# Run Vina locally
vina --receptor receptor.pdbqt --ligand ligand.pdbqt --config config.txt
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | `BioDockify-jobs-use1-1763775915` | S3 bucket for job files |
| `BATCH_JOB_QUEUE` | `cloudvina-fargate-queue` | AWS Batch job queue |
| `BATCH_JOB_DEFINITION` | `cloudvina-job` | AWS Batch job definition |
| `SUPABASE_URL` | - | Supabase database URL |
| `SUPABASE_KEY` | - | Supabase service key |
| `MD_LAMBDA_FUNCTION` | `MD_Stability_Scorer` | Lambda function for MD scoring |

## Getting Help

If you're still having issues:

1. **Check the logs:**
   - API logs: Check your application logs
   - AWS Batch logs: CloudWatch Logs → `/aws/batch/job`
   - S3 logs: Check S3 server access logs (if enabled)

2. **Review documentation:**
   - [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
   - [`AWS_SETUP.md`](AWS_SETUP.md)
   - [`README.md`](README.md)

3. **Check recent changes:**
   ```bash
   git log --oneline -20
   git diff HEAD~1
   ```

4. **Run the debug script again:**
   ```bash
   python debug_docking_jobs.py
   ```

5. **Check GitHub Issues:**
   - Search for similar issues: https://github.com/tajo9128/cloudvina/issues

## Performance Tuning

### For Faster Jobs

1. **Reduce exhaustiveness:**
   ```python
   grid_params = {'exhaustiveness': 4}  # Default: 16
   ```

2. **Use smaller search box:**
   ```python
   grid_params = {
       'size_x': 18.0,
       'size_y': 18.0,
       'size_z': 18.0
   }
   ```

3. **Use Vina only (skip consensus):**
   ```python
   engine = 'vina'  # Instead of 'consensus'
   ```

### For Better Accuracy

1. **Increase exhaustiveness:**
   ```python
   grid_params = {'exhaustiveness': 32}  # Default: 16
   ```

2. **Use consensus engine:**
   ```python
   engine = 'consensus'  # Runs Vina + RF + Gnina
   ```

3. **Enable ensemble mode:**
   ```python
   config = {
       'ensemble_mode': True,
       'engine': 'consensus'
   }
   ```

## Security Considerations

1. **Never commit credentials:**
   - Use `.env` files (already in `.gitignore`)
   - Use AWS IAM roles when possible
   - Rotate access keys regularly

2. **Limit S3 bucket access:**
   - Use bucket policies
   - Enable versioning
   - Enable encryption

3. **Monitor AWS costs:**
   - Set up billing alerts
   - Monitor Batch job usage
   - Clean up old S3 files regularly

## Additional Resources

- [AutoDock Vina Documentation](http://vina.scripps.edu/)
- [Gnina Documentation](https://github.com/gnina/gnina)
- [AWS Batch Documentation](https://docs.aws.amazon.com/batch/)
- [Supabase Documentation](https://supabase.com/docs)

---

**Last Updated:** 2026-01-13
**Version:** 1.0
