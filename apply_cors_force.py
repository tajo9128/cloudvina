import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Target Bucket from api/aws_services.py
BUCKET_NAME = "cloudvina-jobs-use1-1763775915"

def apply_cors():
    print(f"Applying CORS to bucket: {BUCKET_NAME}...")
    s3 = boto3.client('s3')
    
    cors_configuration = {
        'CORSRules': [{
            'AllowedHeaders': ['*'],
            'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
            'AllowedOrigins': ['*'],
            'ExposeHeaders': ['ETag', 'x-amz-server-side-encryption', 'x-amz-request-id', 'x-amz-id-2'],
            'MaxAgeSeconds': 3000
        }]
    }

    try:
        s3.put_bucket_cors(Bucket=BUCKET_NAME, CORSConfiguration=cors_configuration)
        print("✅ Success! CORS policy applied.")
        print("You can verify by refreshing the results page.")
    except Exception as e:
        print(f"❌ Failed to set CORS: {e}")

if __name__ == "__main__":
    apply_cors()
