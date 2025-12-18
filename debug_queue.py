import boto3
import os

def check_batch_status():
    print("Initializing Boto3 Batch Client...")
    try:
        # Use default region or env var
        region = os.getenv('AWS_REGION', 'us-east-1')
        print(f"Region: {region}")
        batch = boto3.client('batch', region_name=region)
        
        print("\n--- JOB QUEUES ---")
        queues = batch.describe_job_queues()
        if not queues['jobQueues']:
            print("No Job Queues found.")
        
        for q in queues['jobQueues']:
            print(f"Queue Name: {q['jobQueueName']}")
            print(f"  ARN: {q['jobQueueArn']}")
            print(f"  State: {q['state']} (Should be ENABLED)") 
            print(f"  Status: {q['status']} (Should be VALID)") 
            if 'statusReason' in q and q['statusReason']:
                print(f"  Reason: {q['statusReason']}")
            
            # Check attached CEs
            print("  Compute Environments:")
            for ce_order in q['computeEnvironmentOrder']:
                ce_arn = ce_order['computeEnvironment']
                
                # Check CE status
                try:
                    ce_resp = batch.describe_compute_environments(computeEnvironments=[ce_arn])
                    if ce_resp['computeEnvironments']:
                        ce = ce_resp['computeEnvironments'][0]
                        print(f"    -> {ce['computeEnvironmentName']}")
                        print(f"       State: {ce['state']} (Should be ENABLED)")
                        print(f"       Status: {ce['status']} (Should be VALID)")
                        if 'statusReason' in ce and ce['statusReason']:
                            print(f"       Reason: {ce['statusReason']}")
                    else:
                        print(f"    -> {ce_arn} (NOT FOUND)")
                except Exception as ce_err:
                    print(f"    -> Error fetching CE {ce_arn}: {ce_err}")

    except Exception as e:
        print(f"Error checking batch: {e}")
        print("Hint: Check if AWS credentials are set in environment or ~/.aws/credentials")

if __name__ == "__main__":
    check_batch_status()
