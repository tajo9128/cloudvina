"""
AWS services integration (S3 and Batch)
"""
import boto3
from botocore.exceptions import ClientError
import os
from typing import Tuple

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "cloudvina-jobs-use1-1763775915")
BATCH_JOB_QUEUE = os.getenv("BATCH_JOB_QUEUE", "cloudvina-fargate-queue")
BATCH_JOB_DEFINITION = os.getenv("BATCH_JOB_DEFINITION", "cloudvina-fargate-job-v10")

# Initialize clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
batch_client = boto3.client('batch', region_name=AWS_REGION)


def generate_presigned_upload_urls(job_id: str, receptor_filename: str, ligand_filename: str) -> Tuple[str, str, str, str]:
    """
    Generate pre-signed URLs for uploading receptor and ligand files to S3
    
    Returns:
        Tuple of (receptor_url, ligand_url, receptor_key, ligand_key)
    """
    # Determine extensions
    receptor_ext = os.path.splitext(receptor_filename)[1].lower() or '.pdb'
    ligand_ext = os.path.splitext(ligand_filename)[1].lower() or '.pdbqt'
    
    # Construct keys
    receptor_key = f"jobs/{job_id}/receptor_input{receptor_ext}"
    ligand_key = f"jobs/{job_id}/ligand_input{ligand_ext}"
    
    # Determine content types
    content_types = {
        '.pdb': 'chemical/x-pdb',
        '.pdbqt': 'chemical/x-pdbqt',
        '.sdf': 'chemical/x-mdl-sdfile',
        '.mol2': 'chemical/x-mol2'
    }
    
    receptor_type = content_types.get(receptor_ext, 'application/octet-stream')
    ligand_type = content_types.get(ligand_ext, 'application/octet-stream')
    
    try:
        receptor_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': receptor_key,
                'ContentType': receptor_type
            },
            ExpiresIn=300  # 5 minutes
        )
        
        ligand_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': ligand_key,
                'ContentType': ligand_type
            },
            ExpiresIn=300
        )
        
        return receptor_url, ligand_url, receptor_key, ligand_key
    
    except ClientError as e:
        raise Exception(f"Failed to generate presigned URLs: {str(e)}")


def generate_presigned_download_url(job_id: str, filename: str) -> str:
    """
    Generate pre-signed URL for downloading result file from S3
    """
    key = f"jobs/{job_id}/{filename}"
    
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': key},
            ExpiresIn=3600  # 1 hour
        )
        return url
    
    except ClientError as e:
        raise Exception(f"Failed to generate download URL: {str(e)}")


def submit_batch_job(job_id: str, receptor_key: str, ligand_key: str) -> str:
    """
    Submit job to AWS Batch
    
    Returns:
        AWS Batch job ID
    """
    try:
        response = batch_client.submit_job(
            jobName=f'cloudvina-{job_id}',
            jobQueue=BATCH_JOB_QUEUE,
            jobDefinition=BATCH_JOB_DEFINITION,
            containerOverrides={
                'environment': [
                    {'name': 'JOB_ID', 'value': job_id},
                    {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                    {'name': 'RECEPTOR_S3_KEY', 'value': receptor_key},
                    {'name': 'LIGAND_S3_KEY', 'value': ligand_key}
                ]
            }
        )
        
        return response['jobId']
    
    except ClientError as e:
        raise Exception(f"Failed to submit Batch job: {str(e)}")


def get_batch_job_status(batch_job_id: str) -> dict:
    """
    Get status of AWS Batch job
    
    Returns:
        Dict with status and details
    """
    try:
        response = batch_client.describe_jobs(jobs=[batch_job_id])
        
        if not response['jobs']:
            raise Exception(f"Batch job {batch_job_id} not found")
        
        job = response['jobs'][0]
        
        return {
            'status': job['status'],
            'status_reason': job.get('statusReason'),
            'created_at': job.get('createdAt'),
            'started_at': job.get('startedAt'),
            'stopped_at': job.get('stoppedAt')
        }
    
    except ClientError as e:
        raise Exception(f"Failed to get Batch job status: {str(e)}")
