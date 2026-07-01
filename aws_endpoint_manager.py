import os
import sys
import argparse
import time
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::891377369931:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# Resource Names
MODEL_NAME = "dwp-cmg-llama-8b-model-v2"
ENDPOINT_CONFIG_NAME = "dwp-cmg-llama-8b-config-v2"
ENDPOINT_NAME = "dwp-cmg-llama-8b-endpoint-v2"

def get_sagemaker_client():
    return boto3.client("sagemaker", region_name=REGION)

def get_cloudwatch_client():
    return boto3.client("cloudwatch", region_name=REGION)

def deploy_endpoint(model_s3_uri, instance_type="ml.g5.2xlarge"):
    sm = get_sagemaker_client()
    
    # Clean up existing resources if they exist
    delete_endpoint()
    
    # 1. Retrieve Hugging Face Inference TGI container URI
    # TGI container optimized for LLM serving
    print("Retrieving Hugging Face Inference TGI container Image URI...")
    try:
        import sagemaker
        container_uri = sagemaker.image_uris.retrieve(
            framework="huggingface",
            region=REGION,
            version="2.4.0",
            image_scope="inference",
            instance_type=instance_type
        )
    except Exception:
        # Fallback to hardcoded verified TGI container URI in us-east-1 (TGI 3.0.1 natively supports Llama 3.1)
        container_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi3.0.1-gpu-py311-cu124-ubuntu22.04"
        
    print(f"Container URI: {container_uri}")
    
    # 2. Create Model
    print(f"Creating SageMaker Model '{MODEL_NAME}' pointing to {model_s3_uri}...")
    try:
        sm.create_model(
            ModelName=MODEL_NAME,
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer={
                "Image": container_uri,
                "ModelDataUrl": model_s3_uri,
                "Environment": {
                    "HF_MODEL_ID": "/opt/ml/model",
                    "MAX_INPUT_LENGTH": "1024",
                    "MAX_TOTAL_TOKENS": "2048",
                    "NUMBER_OF_GPU": "1",
                    "MESSAGES_API_ENABLED": "true"  # Enables OpenAI-compatible messages API endpoint in TGI
                }
            }
        )
    except ClientError as e:
        print(f"Error creating model: {e}")
        return False

    # 3. Create Endpoint Config
    print(f"Creating Endpoint Configuration '{ENDPOINT_CONFIG_NAME}' on {instance_type}...")
    try:
        sm.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": MODEL_NAME,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                    "InitialVariantWeight": 1.0
                }
            ]
        )
    except ClientError as e:
        print(f"Error creating endpoint configuration: {e}")
        return False

    # 4. Create Endpoint
    print(f"Creating Real-Time Endpoint '{ENDPOINT_NAME}'...")
    try:
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print("Deployment started successfully. Polling status (takes 3-5 minutes)...")
        return wait_for_endpoint()
    except ClientError as e:
        print(f"Error creating endpoint: {e}")
        return False

def wait_for_endpoint():
    sm = get_sagemaker_client()
    while True:
        try:
            desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
            status = desc["EndpointStatus"]
            print(f"Endpoint Status: {status}")
            
            if status == "InService":
                print("Endpoint is online and ready for inference!")
                return True
            elif status in ["Failed", "OutOfService"]:
                print(f"Endpoint creation failed with status: {status}")
                if "FailureReason" in desc:
                    print(f"Reason: {desc['FailureReason']}")
                return False
                
        except ClientError as e:
            print(f"Error polling endpoint status: {e}")
            return False
        time.sleep(15)

def check_status():
    sm = get_sagemaker_client()
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint: {desc['EndpointName']}")
        print(f"Status: {desc['EndpointStatus']}")
        print(f"Config: {desc['EndpointConfigName']}")
        print(f"Created: {desc['CreationTime']}")
        return desc["EndpointStatus"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint '{ENDPOINT_NAME}' does not exist.")
        else:
            print(f"Error describing endpoint: {e}")
        return "NotExist"

def delete_endpoint():
    sm = get_sagemaker_client()
    
    # Delete Endpoint
    endpoint_exists = True
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleting Endpoint '{ENDPOINT_NAME}'...")
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("Endpoint deletion triggered. Waiting for deletion to complete...")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint '{ENDPOINT_NAME}' does not exist.")
            endpoint_exists = False
        else:
            print(f"Error deleting endpoint: {e}")
            
    # Wait for endpoint deletion to complete
    if endpoint_exists:
        while True:
            try:
                desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
                status = desc["EndpointStatus"]
                print(f"Endpoint deletion status: {status}...")
                time.sleep(10)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ValidationException":
                    print("Endpoint deleted successfully.")
                    break
                else:
                    print(f"Error waiting for endpoint deletion: {e}")
                    break
            
    # Delete Endpoint Config
    try:
        print(f"Deleting Endpoint Config '{ENDPOINT_CONFIG_NAME}'...")
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            print(f"Error deleting configuration: {e}")
            
    # Delete Model
    try:
        print(f"Deleting Model '{MODEL_NAME}'...")
        sm.delete_model(ModelName=MODEL_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            print(f"Error deleting model: {e}")
            
    print("Cleanup completed.")


def monitor_and_autodelete(max_idle_minutes=30):
    """
    Checks the number of requests in CloudWatch. If 0 requests occur within the 
    last 'max_idle_minutes', automatically deletes the endpoint to save credits.
    """
    print(f"Starting idle monitor for '{ENDPOINT_NAME}' (Limit: {max_idle_minutes} minutes)...")
    cw = get_cloudwatch_client()
    sm = get_sagemaker_client()
    
    # First verify the endpoint exists and is active
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        if desc["EndpointStatus"] != "InService":
            print("Endpoint is not InService. Exiting monitor.")
            return
    except ClientError:
        print("Endpoint does not exist. Exiting monitor.")
        return

    # Check request count every 5 minutes
    check_interval_seconds = 300
    idle_threshold_seconds = max_idle_minutes * 60
    
    while True:
        try:
            # Query CloudWatch metric: Invocations
            # Sum of invocations over the last max_idle_minutes period
            end_time = time.time()
            start_time = end_time - idle_threshold_seconds
            
            response = cw.get_metric_data(
                MetricDataQueries=[
                    {
                        "Id": "invocations",
                        "MetricStat": {
                            "Metric": {
                                "Namespace": "AWS/SageMaker",
                                "MetricName": "Invocations",
                                "Dimensions": [
                                    {
                                        "Name": "EndpointName",
                                        "Value": ENDPOINT_NAME
                                    },
                                    {
                                        "Name": "VariantName",
                                        "Value": "AllTraffic"
                                    }
                                ]
                            },
                            "Period": idle_threshold_seconds,
                            "Stat": "Sum"
                        }
                    }
                ],
                StartTime=time.fromtimestamp(start_time),
                EndTime=time.fromtimestamp(end_time)
            )
            
            metric_results = response["MetricDataResults"][0]
            values = metric_results.get("Values", [])
            
            # If values is empty or sum is 0, endpoint is idle
            total_requests = sum(values) if values else 0.0
            print(f"Monitor check: Invocations in the last {max_idle_minutes}m = {total_requests}")
            
            if total_requests == 0.0:
                print(f"ENDPOINT IS IDLE FOR {max_idle_minutes} MINUTES. TRIGGERING AUTO-DELETE TO CONSERVE CREDITS!")
                delete_endpoint()
                break
                
        except Exception as e:
            print(f"Error querying monitor metrics: {e}")
            
        time.sleep(check_interval_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Endpoint Manager")
    parser.add_argument("action", choices=["deploy", "status", "delete", "monitor"])
    parser.add_argument("--model_uri", help="S3 URI to model.tar.gz (required for deploy)")
    parser.add_argument("--instance_type", default="ml.g5.2xlarge", help="SageMaker GPU instance type to use for deployment")
    parser.add_argument("--idle_minutes", type=int, default=30, help="Idle timeout minutes for monitor")
    args = parser.parse_args()

    if args.action == "deploy":
        if not args.model_uri:
            print("Error: --model_uri is required for deploy.")
            sys.exit(1)
        deploy_endpoint(args.model_uri, args.instance_type)
    elif args.action == "status":
        check_status()
    elif args.action == "delete":
        delete_endpoint()
    elif args.action == "monitor":
        monitor_and_autodelete(args.idle_minutes)
