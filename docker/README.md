# CloudVina Docker Container

This Docker container runs AutoDock Vina molecular docking simulations in an isolated, cloud-ready environment.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t cloudvina:latest .
```

### 2. Test Locally (without AWS)

For local testing without S3, you can mount files directly:

```bash
docker run --rm \
  -v $(pwd)/test_data:/app/work \
  -e JOB_ID=test-001 \
  -e RECEPTOR_S3_KEY=receptor.pdb \
  -e LIGAND_S3_KEY=ligand.sdf \
  cloudvina:latest
```

### 3. Run with AWS S3 (Production Mode)

```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e JOB_ID=job-12345 \
  -e S3_BUCKET=cloudvina-jobs \
  -e RECEPTOR_S3_KEY=uploads/receptor.pdb \
  -e LIGAND_S3_KEY=uploads/ligand.sdf \
  cloudvina:latest
```

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `JOB_ID` | Yes | Unique job identifier | `job-abc123` |
| `S3_BUCKET` | No | S3 bucket name | `cloudvina-jobs` (default) |
| `RECEPTOR_S3_KEY` | Yes | S3 path to receptor file | `uploads/1a2b.pdb` |
| `LIGAND_S3_KEY` | Yes | S3 path to ligand file | `uploads/ligand.sdf` |
| `AWS_ACCESS_KEY_ID` | Yes* | AWS credentials | - |
| `AWS_SECRET_ACCESS_KEY` | Yes* | AWS credentials | - |

*Not required if using IAM roles (recommended for AWS Batch)

## What This Container Does

1. **Downloads** receptor and ligand files from S3
2. **Converts** ligand to PDBQT format using OpenBabel
3. **Prepares** receptor (assumes pre-prepared PDBQT or converts PDB)
4. **Runs** AutoDock Vina with default grid box (20×20×20 Å)
5. **Uploads** results (`output.pdbqt`, `log.txt`) back to S3

## Output Files

Results are uploaded to: `s3://[bucket]/jobs/[JOB_ID]/`

- `output.pdbqt` - Docked ligand poses
- `log.txt` - Vina log with binding affinities
- `SUCCESS` - Marker file indicating successful completion

## Testing

### Prepare Test Data

```bash
mkdir test_data
cd test_data

# Download sample files (example: HIV protease)
wget https://files.rcsb.org/download/1HVR.pdb -O receptor.pdb
# Download a sample ligand (you'll need to provide this)
```

### Build and Test

```bash
# Build
docker build -t cloudvina:latest .

# Run test
docker run --rm \
  -v $(pwd)/test_data:/app/work \
  cloudvina:latest
```

## Pushing to AWS ECR

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag cloudvina:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/cloudvina:latest

# Push
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/cloudvina:latest
```

## Troubleshooting

**Error: "Missing required environment variables"**
- Ensure `JOB_ID`, `RECEPTOR_S3_KEY`, and `LIGAND_S3_KEY` are set

**Error: "Access Denied" from S3**
- Check AWS credentials
- Verify S3 bucket permissions
- For AWS Batch, ensure the job role has S3 read/write access

**Error: "Vina failed"**
- Check that input files are valid PDB/SDF format
- Verify ligand has explicit hydrogens
- Review the log output for Vina-specific errors

## Performance

- **t2.micro** (Free Tier): ~30-45 mins per dock
- **c6i.large** (Spot): ~5-10 mins per dock

## Next Steps

After testing locally:
1. Push image to AWS ECR
2. Create AWS Batch job definition
3. Test on AWS Batch with Free Tier `t2.micro`
4. Scale to Spot Instances for production
