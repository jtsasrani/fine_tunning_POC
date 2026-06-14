import os
import re
import json
import ast

def parse_log(log_path):
    train_history = []
    eval_history = []
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} does not exist.")
        return train_history, eval_history
        
    # Regex to match python dict output of HF trainer
    # Example: {'loss': '1.011', 'grad_norm': '0.4975', 'learning_rate': '9.305e-05', 'epoch': '0.6363'}
    # Example: {'eval_loss': '0.8885', 'eval_runtime': '425.8', ...}
    dict_pattern = re.compile(r"\{'[a-z_]+':\s*.*\}")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = dict_pattern.search(line)
            if match:
                try:
                    # Safely evaluate python dict string
                    data = ast.literal_eval(match.group(0))
                    
                    if 'loss' in data:
                        train_history.append({
                            'step': len(train_history) * 10 + 10, # logging_steps=10
                            'loss': float(data['loss']),
                            'grad_norm': float(data.get('grad_norm', 0.0)),
                            'lr': float(data['learning_rate']),
                            'epoch': float(data['epoch'])
                        })
                    elif 'eval_loss' in data:
                        # Find the matching step based on epoch
                        epoch = float(data['epoch'])
                        # If we evaluate every 100 steps, step is approx epoch * (total_steps/total_epochs)
                        # We can estimate step from epoch
                        step = round(epoch * (2406 / 3.0))
                        eval_history.append({
                            'step': step,
                            'eval_loss': float(data['eval_loss']),
                            'eval_runtime': float(data.get('eval_runtime', 0.0)),
                            'epoch': epoch
                        })
                except Exception as e:
                    # Skip invalid lines
                    pass
                    
    return train_history, eval_history

def generate_ascii_plot(train_hist, eval_hist):
    # Plot loss curve in ASCII
    if not train_hist:
        return "No training history available to plot."
        
    steps = [x['step'] for x in train_hist]
    losses = [x['loss'] for x in train_hist]
    
    eval_steps = [x['step'] for x in eval_hist]
    eval_losses = [x['eval_loss'] for x in eval_hist]
    
    # Grid size
    width = 60
    height = 15
    
    min_step, max_step = min(steps), max(steps)
    min_loss = min(losses + eval_losses)
    max_loss = max(losses + eval_losses)
    
    # Add a bit of padding to loss range
    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0
    min_loss -= 0.05 * loss_range
    max_loss += 0.05 * loss_range
    
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Helper to map values to grid coordinates
    def get_coords(step, loss):
        x = int((step - min_step) / (max_step - min_step) * (width - 1)) if max_step != min_step else 0
        y = int((loss - min_loss) / (max_loss - min_loss) * (height - 1)) if max_loss != min_loss else 0
        # Invert y because index 0 is top of terminal
        return x, height - 1 - y
        
    # Plot training loss ('.' marks)
    for step, loss in zip(steps, losses):
        x, y = get_coords(step, loss)
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '.'
            
    # Plot eval loss ('*' marks, taking precedence)
    for step, loss in zip(eval_steps, eval_losses):
        x, y = get_coords(step, loss)
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '★'
            
    # Compile text plot
    plot_lines = []
    for r in range(height):
        # Y-axis labels
        y_val = max_loss - r * (max_loss - min_loss) / (height - 1)
        row_str = f"{y_val:6.3f} | " + "".join(grid[r])
        plot_lines.append(row_str)
        
    # X-axis
    plot_lines.append("       +" + "-" * width)
    # X-axis labels
    label_line = f"        {min_step}" + " " * (width - len(str(min_step)) - len(str(max_step)) - 2) + f"{max_step} (Steps)"
    plot_lines.append(label_line)
    
    legend = "\nLegend:  . = Training Loss    ★ = Validation Loss"
    return "\n".join(plot_lines) + legend

def main():
    log_path = "data/pipeline_step6_training.log"
    report_path = "data/training_stats_report.md"
    
    print(f"Reading logs from {log_path}...")
    train_hist, eval_hist = parse_log(log_path)
    
    if not train_hist:
        print("No training records found. Exiting.")
        return
        
    print(f"Parsed {len(train_hist)} training logging points and {len(eval_hist)} validation evaluations.")
    
    # Compute summary stats
    total_steps = 2406
    completed_steps = train_hist[-1]['step'] if train_hist else 0
    progress_pct = (completed_steps / total_steps) * 100
    
    initial_loss = train_hist[0]['loss']
    final_loss = train_hist[-1]['loss']
    loss_reduction = initial_loss - final_loss
    
    final_eval_loss = eval_hist[-1]['eval_loss'] if eval_hist else "N/A"
    
    # Generate ASCII plot
    ascii_curve = generate_ascii_plot(train_hist, eval_hist)
    
    # Format report
    lines = [
        "# DWP CMS Student Model Training Report",
        f"\nGenerated automatically at completion/monitoring checkpoints.",
        "\n## Training Progress Summary\n",
        f"* **Model Name**: `Qwen/Qwen2.5-14B-Instruct` (QLoRA 4-bit)",
        f"* **Total Dataset Size**: 14,251 QA pairs (12,825 train / 1,426 validation)",
        f"* **Effective Batch Size**: 16 (per-device batch=1, grad_accum=16)",
        f"* **Progress**: {completed_steps} / {total_steps} steps ({progress_pct:.1f}% completed)",
        f"* **Initial Training Loss**: `{initial_loss:.4f}`",
        f"* **Current/Final Training Loss**: `{final_loss:.4f}` (reduction of `{loss_reduction:.4f}`)",
        f"* **Current/Final Validation Loss**: `{final_eval_loss}`",
        "\n## Loss Convergence Curve\n",
        "```text",
        ascii_curve,
        "```",
        "\n## Validation Checkpoints\n",
        "| Step | Epoch | Validation Loss | Runtime (seconds) | Samples/sec |",
        "|---|---|---|---|---|",
    ]
    
    for ev in eval_hist:
        samples_sec = 1426 / ev['eval_runtime'] if ev['eval_runtime'] > 0 else 0.0
        lines.append(f"| {ev['step']} | {ev['epoch']:.4f} | `{ev['eval_loss']:.4f}` | {ev['eval_runtime']:.1f}s | {samples_sec:.2f} |")
        
    # Append recent training steps
    lines.append("\n## Latest Training Steps History (Logging Interval = 10 steps)\n")
    lines.append("| Step | Epoch | Learning Rate | Training Loss | Grad Norm |")
    lines.append("|---|---|---|---|---|")
    
    # Sample last 20 records
    recent_records = train_hist[-20:]
    for r in recent_records:
        lines.append(f"| {r['step']} | {r['epoch']:.4f} | `{r['lr']:.4e}` | `{r['loss']:.4f}` | `{r['grad_norm']:.4f}` |")
        
    # Save markdown report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"Report successfully compiled and saved to {report_path}")

if __name__ == "__main__":
    main()
