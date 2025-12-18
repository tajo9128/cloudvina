"""
AWS Batch Configuration Validator
Run this to verify batch setup and identify issues
"""
import boto3
import os
from botocore.exceptions import ClientError

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BATCH_JOB_QUEUE = os.getenv("BATCH_JOB_QUEUE", "cloudvina-fargate-queue")
BATCH_JOB_DEFINITION = os.getenv("BATCH_JOB_DEFINITION", "cloudvina-fargate-job-v10")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")

print("="*60)
print("AWS BATCH CONFIGURATION VALIDATOR")
print("="*60)

# Initialize clients
batch = boto3.client('batch', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)
sts = boto3.client('sts', region_name=AWS_REGION)

# 1. Check AWS Identity
print("\n1. AWS Identity:")
try:
    identity = sts.get_caller_identity()
    print(f"   ✓ Account: {identity['Account']}")
    print(f"   ✓ User ARN: {identity['Arn']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# 2. Check S3 Bucket
print(f"\n2. S3 Bucket ({S3_BUCKET}):")
try:
    s3.head_bucket(Bucket=S3_BUCKET)
    print(f"   ✓ Bucket exists and is accessible")
except ClientError as e:
    print(f"   ✗ Failed: {e}")

# 3. Check Batch Job Queue
print(f"\n3. Batch Job Queue ({BATCH_JOB_QUEUE}):")
try:
    response = batch.describe_job_queues(jobQueues=[BATCH_JOB_QUEUE])
    if response['jobQueues']:
        queue = response['jobQueues'][0]
        print(f"   ✓ Queue exists")
        print(f"   ✓ State: {queue['state']}")
        print(f"   ✓ Status: {queue['status']}")
        if queue['state'] != 'ENABLED':
            print(f"   ✗ WARNING: Queue is not ENABLED!")
    else:
        print(f"   ✗ Queue does NOT exist!")
except ClientError as e:
    print(f"   ✗ Failed: {e}")

# 4. Check Job Definition
print(f"\n4. Batch Job Definition ({BATCH_JOB_DEFINITION}):")
try:
    response = batch.describe_job_definitions(
        jobDefinitionName=BATCH_JOB_DEFINITION.split(':')[0],
        status='ACTIVE'
    )
    if response['jobDefinitions']:
        job_defs = sorted(response['jobDefinitions'], key=lambda x: x['revision'], reverse=True)
        latest = job_defs[0]
        print(f"   ✓ Definition exists")
        print(f"   ✓ Latest Revision: {latest['revision']}")
        print(f"   ✓ Full ARN: {latest['jobDefinitionArn']}")
        print(f"   ✓ Container Image: {latest['containerProperties'].get('image', 'N/A')}")
        print(f"   ✓ vCPUs: {latest['containerProperties'].get('vcpus', 'N/A')}")
        print(f"   ✓ Memory: {latest['containerProperties'].get('memory', 'N/A')} MB")
        
        # Check if using correct versioned definition
        if ':' not in BATCH_JOB_DEFINITION:
            print(f"   ⚠ WARNING: Using unversioned definition (will use latest)")
    else:
        print(f"   ✗ Definition does NOT exist!")
except ClientError as e:
    print(f"   ✗ Failed: {e}")

# 5. Test Job Submission (DRY RUN - comments for safety)
print(f"\n5. Test Job Submission (Simulated):")
print(f"   Would submit to: {BATCH_JOB_QUEUE}")
print(f"   Would use definition: {BATCH_JOB_DEFINITION}")
print(f"   Container would receive:")
print(f"     - JOB_ID, S3_BUCKET, RECEPTOR_S3_KEY, LIGAND_S3_KEY, DOCKING_ENGINE")

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
print("\nIf any checks failed (✗), those are the issues to fix!")
