import boto3
import os

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "cloudvina-jobs-use1-1763775915")

def configure_cors():
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    cors_configuration = {
        'CORSRules': [{
            'AllowedHeaders': ['*'],
            'AllowedMethods': ['GET', 'PUT', 'POST', 'HEAD'],
            'AllowedOrigins': ['*'],  # Allow all origins for now
            'ExposeHeaders': ['ETag'],
            'MaxAgeSeconds': 3000
        }]
    }

    try:
        s3.put_bucket_cors(Bucket=S3_BUCKET, CORSConfiguration=cors_configuration)
        print(f"Successfully configured CORS for bucket: {S3_BUCKET}")
    except Exception as e:
        print(f"Error configuring CORS: {e}")

if __name__ == "__main__":
    configure_cors()
