#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioDockify Docking Job Debugging Script

This script helps diagnose issues with docking jobs by checking:
1. Environment configuration
2. AWS credentials and connectivity
3. S3 bucket access
4. Batch job queue and definition
5. Recent job status and errors
6. Common configuration issues
"""

import os
import sys
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime, timedelta
import traceback

# Fix Windows encoding issue
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def check_environment():
    """Check environment variables"""
    print_header("1. ENVIRONMENT CONFIGURATION")
    
    required_vars = [
        'AWS_REGION',
        'S3_BUCKET',
        'BATCH_JOB_QUEUE',
        'BATCH_JOB_DEFINITION',
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]
    
    optional_vars = [
        'MD_LAMBDA_FUNCTION'
    ]
    
    all_good = True
    
    print(f"\n{Colors.BOLD}Required Variables:{Colors.ENDC}")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var}: {value}")
        else:
            print_error(f"{var}: NOT SET")
            all_good = False
    
    print(f"\n{Colors.BOLD}Optional Variables:{Colors.ENDC}")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var}: {value}")
        else:
            print_warning(f"{var}: Not set (optional)")
    
    return all_good

def check_aws_credentials():
    """Check AWS credentials"""
    print_header("2. AWS CREDENTIALS & CONNECTIVITY")
    
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print_success("AWS credentials found")
        print_info(f"Account ID: {identity['Account']}")
        print_info(f"User ARN: {identity['Arn']}")
        
        return True
    except NoCredentialsError:
        print_error("No AWS credentials found")
        print_info("Please configure AWS credentials using:")
        print_info("  - AWS CLI: aws configure")
        print_info("  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return False
    except Exception as e:
        print_error(f"Error checking AWS credentials: {e}")
        return False

def check_s3_access():
    """Check S3 bucket access"""
    print_header("3. S3 BUCKET ACCESS")
    
    bucket_name = os.getenv('S3_BUCKET', 'BioDockify-jobs-use1-1763775915')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print_info(f"Bucket: {bucket_name}")
    print_info(f"Region: {region}\n")
    
    try:
        s3 = boto3.client('s3', region_name=region)
        
        # Check if bucket exists
        s3.head_bucket(Bucket=bucket_name)
        print_success(f"Bucket '{bucket_name}' exists and is accessible")
        
        # Check bucket location
        location = s3.get_bucket_location(Bucket=bucket_name)
        print_info(f"Bucket location: {location['LocationConstraint'] or 'us-east-1'}")
        
        # List recent job folders
        print_info("\nRecent job folders:")
        try:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix='jobs/',
                Delimiter='/',
                MaxKeys=10
            )
            
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes'][:5]:
                    folder = prefix['Prefix'].rstrip('/')
                    print_info(f"  - {folder}")
            else:
                print_warning("No job folders found")
        except Exception as e:
            print_warning(f"Could not list job folders: {e}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print_error(f"Bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            print_error(f"Access denied to bucket '{bucket_name}'")
        else:
            print_error(f"S3 error: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def check_batch_configuration():
    """Check AWS Batch configuration"""
    print_header("4. AWS BATCH CONFIGURATION")
    
    queue_name = os.getenv('BATCH_JOB_QUEUE', 'cloudvina-fargate-queue')
    job_def = os.getenv('BATCH_JOB_DEFINITION', 'cloudvina-job')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print_info(f"Job Queue: {queue_name}")
    print_info(f"Job Definition: {job_def}\n")
    
    try:
        batch = boto3.client('batch', region_name=region)
        
        # Check job queue
        print_info("Checking job queue...")
        try:
            queue = batch.describe_job_queues(jobQueues=[queue_name])
            if queue['jobQueues']:
                queue_info = queue['jobQueues'][0]
                print_success(f"Job queue '{queue_name}' exists")
                print_info(f"  Status: {queue_info['status']}")
                print_info(f"  State: {queue_info['state']}")
                
                # Check compute environment
                compute_envs = queue_info.get('computeEnvironmentOrder', [])
                if compute_envs:
                    print_info(f"  Compute environments: {len(compute_envs)}")
            else:
                print_error(f"Job queue '{queue_name}' not found")
                return False
        except ClientError as e:
            print_error(f"Error checking job queue: {e}")
            return False
        
        # Check job definition
        print_info("\nChecking job definition...")
        try:
            # Handle malformed definition (e.g., ":latest")
            if job_def.startswith(":"):
                print_warning(f"Malformed job definition '{job_def}'")
                job_def = f"biodockify-all-fixes{job_def}"
                print_info(f"Trying corrected definition: {job_def}")
            
            defs = batch.describe_job_definitions(jobDefinitions=[job_def])
            if defs['jobDefinitions']:
                def_info = defs['jobDefinitions'][0]
                print_success(f"Job definition '{job_def}' exists")
                print_info(f"  Status: {def_info['status']}")
                print_info(f"  Type: {def_info['type']}")
                print_info(f"  Container image: {def_info['containerProperties']['image']}")
            else:
                print_error(f"Job definition '{job_def}' not found")
                return False
        except ClientError as e:
            print_error(f"Error checking job definition: {e}")
            return False
        
        return True
        
    except ClientError as e:
        print_error(f"AWS Batch error: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def check_recent_batch_jobs():
    """Check recent Batch job status"""
    print_header("5. RECENT BATCH JOBS")
    
    region = os.getenv('AWS_REGION', 'us-east-1')
    queue_name = os.getenv('BATCH_JOB_QUEUE', 'cloudvina-fargate-queue')
    
    try:
        batch = boto3.client('batch', region_name=region)
        
        # List recent jobs
        print_info(f"Fetching recent jobs from queue '{queue_name}'...")
        response = batch.list_jobs(
            jobQueue=queue_name,
            maxResults=10,
            filters=[
                {'name': 'BEFORE_CREATED_AT', 'values': [(datetime.utcnow() + timedelta(hours=24)).isoformat()]}
            ]
        )
        
        jobs = response.get('jobSummaryList', [])
        
        if not jobs:
            print_warning("No recent jobs found in the last 24 hours")
            return True
        
        print(f"\nFound {len(jobs)} recent job(s):\n")
        
        # Count by status
        status_counts = {}
        for job in jobs:
            status = job['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"{Colors.BOLD}Status Summary:{Colors.ENDC}")
        for status, count in sorted(status_counts.items()):
            color = Colors.OKGREEN if status == 'SUCCEEDED' else Colors.FAIL if status == 'FAILED' else Colors.WARNING
            print(f"  {color}{status}: {count}{Colors.ENDC}")
        
        # Show details of failed jobs
        failed_jobs = [j for j in jobs if j['status'] == 'FAILED']
        if failed_jobs:
            print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Jobs Details:{Colors.ENDC}")
            for job in failed_jobs[:3]:  # Show first 3
                print(f"\n  Job ID: {job['jobId']}")
                print(f"  Name: {job['jobName']}")
                print(f"  Created: {job['createdAt']}")
                
                # Get detailed status
                try:
                    details = batch.describe_jobs(jobs=[job['jobId']])
                    if details['jobs']:
                        job_detail = details['jobs'][0]
                        status_reason = job_detail.get('statusReason', 'No reason provided')
                        print(f"  Status Reason: {status_reason}")
                        
                        # Show container error if available
                        if 'container' in job_detail:
                            container = job_detail['container']
                            if 'command' in container:
                                print(f"  Command: {' '.join(container['command'])}")
                except Exception as e:
                    print(f"  Error getting details: {e}")
        
        return True
        
    except ClientError as e:
        print_error(f"Error listing batch jobs: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def check_common_issues():
    """Check for common configuration issues"""
    print_header("6. COMMON ISSUES CHECK")
    
    issues_found = False
    
    # Check 1: S3 bucket name mismatch
    s3_bucket = os.getenv('S3_BUCKET')
    if s3_bucket:
        if 'cloudvina-jobs' in s3_bucket and 'BioDockify-jobs' not in s3_bucket:
            print_warning("S3_BUCKET uses 'cloudvina-jobs' naming")
            print_info("  This may cause issues if other code expects 'BioDockify-jobs'")
            issues_found = True
    
    # Check 2: Job definition format
    job_def = os.getenv('BATCH_JOB_DEFINITION')
    if job_def and job_def.startswith(":"):
        print_warning("Job definition starts with ':' (malformed)")
        print_info("  This will be auto-corrected in aws_services.py")
        issues_found = True
    
    # Check 3: Default engine
    print_info("Default docking engine: 'consensus'")
    print_info("  This runs Vina -> RF -> Gnina pipeline")
    print_info("  If experiencing issues, try setting engine='vina'")
    
    # Check 4: Autoboxing
    print_info("\nAutoboxing is enabled by default")
    print_info("  Jobs with center=(0,0,0) will use automatic box calculation")
    print_info("  Check receptor PDB files for HETATM ligands")
    
    if not issues_found:
        print_success("No common configuration issues detected")
    
    return True

def generate_diagnostic_report():
    """Generate a summary report"""
    print_header("DIAGNOSTIC SUMMARY")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("\n1. If all checks passed:")
    print("   - Submit a test job and monitor its progress")
    print("   - Check job status using: GET /jobs/{job_id}/status")
    print("   - Review logs in AWS Batch console if job fails")
    
    print("\n2. If AWS credentials failed:")
    print("   - Configure AWS CLI: aws configure")
    print("   - Or set environment variables:")
    print("     export AWS_ACCESS_KEY_ID=your_key")
    print("     export AWS_SECRET_ACCESS_KEY=your_secret")
    
    print("\n3. If S3 access failed:")
    print("   - Verify bucket exists and is in correct region")
    print("   - Check IAM permissions for s3:* operations")
    
    print("\n4. If Batch configuration failed:")
    print("   - Verify job queue and definition exist in AWS Batch")
    print("   - Check compute environment is active")
    print("   - Review job definition container image")
    
    print("\n5. For debugging specific jobs:")
    print("   - Check job status in database")
    print("   - Review S3 bucket for job files")
    print("   - Check AWS Batch job logs")
    print("   - Run this script again after job submission")
    
    print("\n" + "="*70)
    print(f"{Colors.BOLD}For additional help, check:{Colors.ENDC}")
    print("  - TROUBLESHOOTING.md")
    print("  - AWS_SETUP.md")
    print("  - AWS CloudWatch logs for Batch jobs")
    print("="*70 + "\n")

def main():
    """Main debugging function"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("=" * 70)
    print("           BioDockify Docking Job Debugging Tool")
    print("=" * 70)
    print(f"{Colors.ENDC}")
    
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    results = []
    
    # Run all checks
    results.append(("Environment", check_environment()))
    results.append(("AWS Credentials", check_aws_credentials()))
    results.append(("S3 Access", check_s3_access()))
    results.append(("Batch Configuration", check_batch_configuration()))
    results.append(("Recent Batch Jobs", check_recent_batch_jobs()))
    results.append(("Common Issues", check_common_issues()))
    
    # Generate summary
    print_header("CHECK RESULTS SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.OKGREEN}PASSED{Colors.ENDC}" if result else f"{Colors.FAIL}FAILED{Colors.ENDC}"
        print(f"{name:.<50} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} checks passed{Colors.ENDC}")
    
    # Generate diagnostic report
    generate_diagnostic_report()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
