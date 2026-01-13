# BioDockify Docking Job Debugging Tools

This directory contains comprehensive debugging tools for diagnosing and fixing issues with docking jobs in the BioDockify project.

## Quick Start

### Option 1: Automated Debugging Script (Recommended)

Run the comprehensive Python debugging script:

```bash
cd cloudvina
python debug_docking_jobs.py
```

This script will:
- ✅ Check environment configuration
- ✅ Verify AWS credentials and connectivity
- ✅ Test S3 bucket access
- ✅ Validate AWS Batch configuration
- ✅ Check recent job status
- ✅ Identify common configuration issues
- ✅ Generate diagnostic report with next steps

### Option 2: Quick Shell Script

For Linux/Mac users:
```bash
cd cloudvina
chmod +x QUICK_DEBUG_COMMANDS.sh
./QUICK_DEBUG_COMMANDS.sh
```

For Windows users:
```cmd
cd cloudvina
QUICK_DEBUG_COMMANDS.bat
```

This provides a quick overview of your setup status.

## Tools Overview

### 1. [`debug_docking_jobs.py`](debug_docking_jobs.py)

Comprehensive Python debugging script with detailed checks and colored output.

**Features:**
- Environment variable validation
- AWS credentials verification
- S3 bucket access testing
- AWS Batch configuration check
- Recent job status analysis
- Common issues detection
- Detailed diagnostic report

**Usage:**
```bash
python debug_docking_jobs.py
```

**Output:**
- Color-coded results (green = success, yellow = warning, red = error)
- Detailed error messages
- Actionable next steps
- Summary report

### 2. [`QUICK_DEBUG_COMMANDS.sh`](QUICK_DEBUG_COMMANDS.sh)

Quick shell script for Linux/Mac users.

**Features:**
- Fast environment checks
- AWS credentials verification
- S3 bucket listing
- Batch queue status
- Recent jobs overview

**Usage:**
```bash
chmod +x QUICK_DEBUG_COMMANDS.sh
./QUICK_DEBUG_COMMANDS.sh
```

### 3. [`QUICK_DEBUG_COMMANDS.bat`](QUICK_DEBUG_COMMANDS.bat)

Quick batch script for Windows users.

**Features:**
- Same as shell script but for Windows CMD
- Compatible with Windows environment

**Usage:**
```cmd
QUICK_DEBUG_COMMANDS.bat
```

### 4. [`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md)

Comprehensive troubleshooting guide with solutions for common issues.

**Contents:**
- 10+ common issues with detailed solutions
- Manual debugging steps
- Environment variables reference
- Performance tuning tips
- Security considerations
- Additional resources

## Common Issues Addressed

### 1. AWS Credentials Not Found
**Symptoms:** `No AWS credentials found`

**Solution:**
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### 2. S3 Bucket Access Denied
**Symptoms:** `Access denied to bucket`

**Solution:**
- Verify bucket exists
- Check IAM permissions
- Verify bucket region

### 3. Batch Queue Not Found
**Symptoms:** `Job queue not found`

**Solution:**
- List available queues: `aws batch describe-job-queues`
- Update environment variable

### 4. Job Definition Not Found
**Symptoms:** `Job definition not found`

**Solution:**
- List available definitions: `aws batch describe-job-definitions`
- Check for malformed definitions (starting with `:`)

### 5. Jobs Stuck in SUBMITTED/RUNNABLE
**Symptoms:** Jobs remain in SUBMITTED or RUNNABLE state

**Solution:**
- Check compute environment status
- Verify vCPU limits
- Review CloudWatch logs

### 6. Receptor Conversion Failed
**Symptoms:** `Receptor conversion failed: PDBQT file too short`

**Solution:**
- Verify receptor file format
- Check for ATOM records
- Test conversion manually

### 7. Ligand Conversion Failed
**Symptoms:** `Ligand conversion failed: SMILES parsing error`

**Solution:**
- Check ligand file format
- Verify molecule validity
- Test conversion manually

### 8. Autoboxing Issues
**Symptoms:** `No co-crystallized ligand detected`

**Solution:**
- Check for HETATM ligands
- Manually specify grid parameters
- Use different receptor

### 9. Consensus Engine Issues
**Symptoms:** `Consensus: Vina/RF/Gnina failed`

**Solution:**
- Try with simple Vina engine
- Check binary availability
- Verify model services

### 10. Timeout Issues
**Symptoms:** `Gnina timed out after 300 seconds`

**Solution:**
- Increase timeout
- Reduce exhaustiveness
- Use smaller search box

## Manual Debugging Workflow

### Step 1: Check Job Status
```bash
# Using API
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-api.com/jobs/{job_id}/status

# Or check database directly
```

### Step 2: Check AWS Batch Job
```bash
# Get batch job details
aws batch describe-jobs --jobs $BATCH_JOB_ID

# Check CloudWatch logs
aws logs tail /aws/batch/job --follow
```

### Step 3: Check S3 Files
```bash
# List job files
aws s3 ls s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/

# Download and inspect
aws s3 cp s3://BioDockify-jobs-use1-1763775915/jobs/{job_id}/log.txt .
cat log.txt
```

### Step 4: Check Container Logs
```bash
# Get task ARN from batch job
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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | `BioDockify-jobs-use1-1763775915` | S3 bucket for job files |
| `BATCH_JOB_QUEUE` | `cloudvina-fargate-queue` | AWS Batch job queue |
| `BATCH_JOB_DEFINITION` | `cloudvina-job` | AWS Batch job definition |
| `SUPABASE_URL` | - | Supabase database URL |
| `SUPABASE_KEY` | - | Supabase service key |
| `MD_LAMBDA_FUNCTION` | `MD_Stability_Scorer` | Lambda function for MD scoring |

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

3. **Use Vina only:**
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

## Getting Help

If you're still having issues:

1. **Run the debug script:**
   ```bash
   python debug_docking_jobs.py
   ```

2. **Check logs:**
   - API logs: Check your application logs
   - AWS Batch logs: CloudWatch Logs → `/aws/batch/job`
   - S3 logs: Check S3 server access logs (if enabled)

3. **Review documentation:**
   - [`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md)
   - [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
   - [`AWS_SETUP.md`](AWS_SETUP.md)
   - [`README.md`](README.md)

4. **Check recent changes:**
   ```bash
   git log --oneline -20
   git diff HEAD~1
   ```

5. **Check GitHub Issues:**
   - Search for similar issues: https://github.com/tajo9128/cloudvina/issues

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

## Contributing

If you find a bug or have a suggestion for improving these debugging tools:

1. Open an issue on GitHub
2. Describe the problem or suggestion
3. Provide steps to reproduce (if applicable)
4. Include relevant logs or error messages

## License

These debugging tools are part of the BioDockify project and follow the same license.

---

**Last Updated:** 2026-01-13  
**Version:** 1.0  
**Maintainer:** BioDockify Team
