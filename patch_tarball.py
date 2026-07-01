import boto3
import tarfile
import json
import threading
import os
import io

AWS_REGION = "us-east-1"
BUCKET_NAME = "cms-caseworker-support"
SRC_KEY = "out/dwp-cmg-sft-1782729106/output/model.tar.gz"
DST_KEY = "out/dwp-cmg-sft-1782729106/output/model.tar.gz"  # Overwrite

def main():
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    print("Opening source S3 stream...")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=SRC_KEY)
    src_stream = response['Body']
    
    # Create POSIX pipe for streaming
    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, 'rb')
    w_file = os.fdopen(w_fd, 'wb')
    
    def run_patcher():
        try:
            print("Patcher thread started...")
            with tarfile.open(fileobj=src_stream, mode='r|gz') as src_tar:
                with tarfile.open(fileobj=w_file, mode='w|gz') as dst_tar:
                    for member in src_tar:
                        name = member.name.split('/')[-1]
                        if name == "config.json":
                            print(f"Found config.json ({member.name}). Extracting...")
                            f = src_tar.extractfile(member)
                            if f:
                                config_data = json.loads(f.read().decode('utf-8'))
                                if "quantization_config" in config_data:
                                    del config_data["quantization_config"]
                                    print("Successfully removed quantization_config section.")
                                config_data["torch_dtype"] = "float16"
                                print("Successfully changed torch_dtype to float16.")
                                
                                patched_bytes = json.dumps(config_data, indent=2).encode('utf-8')
                                
                                new_member = tarfile.TarInfo(name=member.name)
                                new_member.size = len(patched_bytes)
                                new_member.mtime = member.mtime
                                new_member.mode = member.mode
                                
                                dst_tar.addfile(new_member, io.BytesIO(patched_bytes))
                        else:
                            if member.isfile():
                                f = src_tar.extractfile(member)
                                dst_tar.addfile(member, f)
                            else:
                                dst_tar.addfile(member)
            print("Patcher thread finished successfully.")
        except Exception as e:
            print(f"Patcher thread failed with exception: {e}")
        finally:
            w_file.close()
            
    t = threading.Thread(target=run_patcher)
    t.start()
    
    # Main thread uploads from pipe read-end directly to S3
    print("Starting S3 multipart upload from stream...")
    try:
        s3.upload_fileobj(r_file, BUCKET_NAME, DST_KEY)
        print("S3 upload completed successfully!")
    except Exception as e:
        print(f"S3 upload failed: {e}")
        
    t.join()
    r_file.close()
    print("Tar patching process finished successfully!")

if __name__ == "__main__":
    main()
