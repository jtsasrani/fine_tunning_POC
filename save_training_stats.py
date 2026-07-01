import boto3
import re
import ast
import csv
import os

AWS_REGION = "us-east-1"
JOB_NAME = "dwp-cmg-sft-1782729106"
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
LOG_STREAM = f"{JOB_NAME}/algo-1-1782729176"

def main():
    os.makedirs("data", exist_ok=True)
    print(f"Connecting to CloudWatch Logs in {AWS_REGION}...")
    client = boto3.client("logs", region_name=AWS_REGION)
    
    events = []
    next_token = None
    
    print(f"Fetching log events for stream {LOG_STREAM}...")
    while True:
        params = {
            "logGroupName": LOG_GROUP,
            "logStreamName": LOG_STREAM,
            "startFromHead": True
        }
        if next_token:
            params["nextToken"] = next_token
            
        try:
            response = client.get_log_events(**params)
        except Exception as e:
            print(f"Error fetching log events: {e}")
            break
            
        new_events = response.get("events", [])
        events.extend(new_events)
        
        new_token = response.get("nextForwardToken")
        if new_token == next_token or not new_events:
            break
        next_token = new_token
        
    print(f"Fetched {len(events)} log lines. Parsing stats...")
    
    train_history = []
    eval_history = []
    
    dict_pattern = re.compile(r"\{'loss':\s*.*\}")
    eval_pattern = re.compile(r"\{'eval_loss':\s*.*\}")
    
    for event in events:
        message = event.get("message", "")
        
        # Check for training metrics
        dict_match = dict_pattern.search(message)
        if dict_match:
            try:
                data = ast.literal_eval(dict_match.group(0))
                train_history.append({
                    'step': len(train_history) * 5 + 5,  # logged every 5 steps
                    'loss': float(data['loss']),
                    'epoch': float(data['epoch']),
                    'learning_rate': float(data['learning_rate'])
                })
            except Exception:
                pass
                
        # Check for validation metrics
        eval_match = eval_pattern.search(message)
        if eval_match:
            try:
                data = ast.literal_eval(eval_match.group(0))
                eval_history.append({
                    'step': len(train_history) * 5,  # aligned to closest step
                    'eval_loss': float(data['eval_loss']),
                    'epoch': float(data['epoch']),
                    'eval_runtime': float(data.get('eval_runtime', 0.0))
                })
            except Exception:
                pass
                
    # Save training history to CSV
    csv_file = "data/training_stats.csv"
    print(f"Saving training loss stats to {csv_file}...")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Step", "Epoch", "Loss", "LearningRate"])
        
        for t in train_history:
            writer.writerow(["Train", t["step"], t["epoch"], t["loss"], t["learning_rate"]])
            
        for e in eval_history:
            writer.writerow(["Eval", e["step"], e["epoch"], e["eval_loss"], ""])
            
    print(f"Finished saving stats! Saved {len(train_history)} training steps and {len(eval_history)} validation checkpoints.")

if __name__ == "__main__":
    main()
