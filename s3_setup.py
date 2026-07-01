import os
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "cms-caseworker-support"
REGION = "us-east-1"

def get_s3_client():
    # Use default profile configuration
    return boto3.client('s3', region_name=REGION)

def verify_bucket_exists():
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' exists and is accessible.")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"Bucket '{BUCKET_NAME}' does not exist. Attempting to create it...")
            try:
                if REGION == 'us-east-1':
                    s3.create_bucket(Bucket=BUCKET_NAME)
                else:
                    s3.create_bucket(
                        Bucket=BUCKET_NAME,
                        CreateBucketConfiguration={'LocationConstraint': REGION}
                    )
                print(f"Bucket '{BUCKET_NAME}' created successfully in {REGION}.")
                return True
            except Exception as create_err:
                print(f"Failed to create bucket: {create_err}")
                return False
        else:
            print(f"Error accessing bucket '{BUCKET_NAME}': {e}")
            return False

def upload_file(local_path, s3_key):
    if not os.path.exists(local_path):
        print(f"Local file '{local_path}' does not exist.")
        return False
        
    s3 = get_s3_client()
    try:
        print(f"Uploading '{local_path}' to 's3://{BUCKET_NAME}/{s3_key}'...")
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print("Upload completed successfully.")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

def download_file(s3_key, local_path):
    s3 = get_s3_client()
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading 's3://{BUCKET_NAME}/{s3_key}' to '{local_path}'...")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print("Download completed successfully.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == '__main__':
    verify_bucket_exists()
