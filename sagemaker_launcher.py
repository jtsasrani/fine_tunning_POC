import os
import argparse
import tarfile
import time
import boto3

# SageMaker Execution Role
ROLE_ARN = "arn:aws:iam::891377369931:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
REGION = "us-east-1"
BUCKET_NAME = "cms-caseworker-support"

# Hugging Face PyTorch Training DLC Image (PyTorch 2.1.0, Transformers 4.36.0, CUDA 12.1)
# Run 2: Standard bfloat16 training (no 4-bit QLoRA) for clean float weights compatible with TGI serving
CONTAINER_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"

def package_and_upload_source():
    s3 = boto3.client('s3', region_name=REGION)
    tar_path = "source.tar.gz"
    
    print("Packaging source files into source.tar.gz...")
    with tarfile.open(tar_path, "w:gz") as tar:
        # Add entrypoint script
        tar.add("sagemaker_entrypoint.py", arcname="sagemaker_entrypoint.py")
        # Add requirements.txt
        if os.path.exists("requirements.txt"):
            tar.add("requirements.txt", arcname="requirements.txt")
        
    s3_key = "source/source.tar.gz"
    print(f"Uploading source archive to s3://{BUCKET_NAME}/{s3_key}...")
    s3.upload_file(tar_path, BUCKET_NAME, s3_key)
    os.remove(tar_path)
    return f"s3://{BUCKET_NAME}/{s3_key}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--on_demand", action="store_true")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token for gated models (e.g. Llama). Can also set HF_TOKEN env var.")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    
    # Generate unique job name
    job_name = f"dwp-cmg-sft-{int(time.time())}"
    
    # Upload source tar.gz to S3
    source_s3_uri = package_and_upload_source()

    # Define hyperparameters matching SageMaker DLC requirements
    hyperparameters = {
        "sagemaker_submit_directory": source_s3_uri,
        "sagemaker_program": "sagemaker_entrypoint.py",
        "sagemaker_container_log_level": "20",
        "sagemaker_region": REGION,
        "model_id": args.model_id,
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "learning_rate": str(args.learning_rate),
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "max_steps": str(args.max_steps)
    }

    # S3 data channels
    train_s3 = f"s3://{BUCKET_NAME}/data/train/"
    val_s3 = f"s3://{BUCKET_NAME}/data/validation/"

    print(f"Configuring SageMaker Training Job: {job_name}...")
    
    # SageMaker Spot instance configurations
    # ml.g5.2xlarge: 1x NVIDIA A10G, 24GB VRAM
    # Spot price saves ~70% over on-demand
    try:
        # Build environment variables for the container
        env_vars = {}
        if args.hf_token:
            env_vars["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
            print("HuggingFace token will be passed to the training container.")
        else:
            print("WARNING: No --hf_token provided. If using a gated model (Llama), download will fail.")

        response = sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": CONTAINER_IMAGE,
                "TrainingInputMode": "File"
            },
            RoleArn=ROLE_ARN,
            Environment=env_vars,
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_s3,
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "application/jsonlines"
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": val_s3,
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "application/jsonlines"
                }
            ],
            OutputDataConfig={
                "S3OutputPath": f"s3://{BUCKET_NAME}/out/"
            },
            ResourceConfig={
                "InstanceType": "ml.g5.2xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": 86400,
                **({"MaxWaitTimeInSeconds": 86400} if not args.on_demand else {})
            },
            HyperParameters=hyperparameters,
            EnableNetworkIsolation=False,
            EnableInterContainerTrafficEncryption=False,
            EnableManagedSpotTraining=not args.on_demand,
            CheckpointConfig={
                "S3Uri": f"s3://{BUCKET_NAME}/checkpoints/"
            }
        )
        
        print(f"\nSageMaker Training Job launched successfully!")
        print(f"Job Name: {job_name}")
        print(f"Monitor command:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region {REGION}")
        
    except Exception as e:
        print(f"Failed to launch training job: {e}")

if __name__ == "__main__":
    main()
