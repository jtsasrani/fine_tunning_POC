import os
import sys
import time
import subprocess

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def check_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def run_command(cmd_list, log_file_path):
    log(f"Running command: {' '.join(cmd_list)}")
    with open(log_file_path, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Real-time logging of subprocess output to the log file and console
        for line in process.stdout:
            log_f.write(line)
            log_f.flush()
            
        return_code = process.wait()
        log(f"Command finished with return code: {return_code}")
        return return_code == 0

def main():
    os.makedirs("data", exist_ok=True)
    
    # Step 1: Wait for Generation Process to complete
    generation_pid = 14570
    log(f"Monitoring remote Q&A generation process (PID: {generation_pid})...")
    
    while True:
        if not check_process_running(generation_pid):
            log("Q&A Generation process has finished (or is no longer running).")
            break
        time.sleep(120)  # Check every 2 minutes
        
    # Check if the generated output exists
    output_raw = "data/llm_generated_training_data.jsonl"
    if not os.path.exists(output_raw) or os.path.getsize(output_raw) == 0:
        log("Error: llm_generated_training_data.jsonl is missing or empty. Aborting pipeline.")
        return
        
    log("Verified generated dataset exists. Proceeding to pipeline execution.")
    
    # Step 2: Run Merge, Deduplicate & Split (Step 5)
    log("=== Launching Step 5: Merge, Deduplicate & Split ===")
    step5_ok = run_command(
        ["/opt/pytorch/bin/python3", "08_merge_training_data.py"],
        "data/pipeline_step5_merge.log"
    )
    if not step5_ok:
        log("Error: Step 5 (Merge & Split) failed. Aborting pipeline.")
        return
        
    # Verify train/val splits exist
    train_split = "data/train_split.jsonl"
    val_split = "data/val_split.jsonl"
    eval_set = "data/evaluation_set.jsonl"
    if not os.path.exists(train_split) or not os.path.exists(val_split):
        log("Error: Train/val splits were not successfully created. Aborting pipeline.")
        return
    log("Deduplication and splitting complete. Datasets created successfully.")
    
    # Step 3: Run Fine-Tuning Student Model (Step 6)
    log("=== Launching Step 6: Fine-Tune Student Model (QLoRA) ===")
    # Fine-tuning will run for 10-14 hours
    step6_ok = run_command(
        ["/opt/pytorch/bin/python3", "02_train_qlora_gpu.py"],
        "data/pipeline_step6_training.log"
    )
    if not step6_ok:
        log("Error: Step 6 (Fine-Tuning) failed. Aborting pipeline.")
        return
        
    # Verify trained merged model exists
    tuned_model_dir = "./trained_models/qwen-14b-cms-qlora_merged"
    if not os.path.exists(tuned_model_dir):
        log(f"Error: Tuned model directory {tuned_model_dir} was not created. Aborting pipeline.")
        return
    log("Fine-tuning completed successfully. Model merged and saved.")
    
    # Step 4: Run Verification & Evaluation (Step 7)
    log("=== Launching Step 7: Verification & Evaluation ===")
    step7_ok = run_command(
        ["/opt/pytorch/bin/python3", "09_evaluate_model.py"],
        "data/pipeline_step7_evaluation.log"
    )
    if not step7_ok:
        log("Error: Step 7 (Evaluation) failed.")
        return
        
    log("=== End-to-End Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
