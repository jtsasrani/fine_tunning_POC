# Implementation Plan - Phase 4: Model Tuning Optimizations

This plan outlines the steps to resolve the style leakage (prepended 5-digit paragraph codes) and repetitive loop behaviors observed in Phase 3 by introducing dataset cleaning, hyperparameter regularization, and nucleus sampling.

## Proposed Changes

### 1. Dataset Cleaning

#### [MODIFY] [01b_ingest_qa_pairs.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/01b_ingest_qa_pairs.py)
We will modify the Q&A ingestion script to clean paragraph numbers from the extracted texts:
* Before saving to the JSONL dataset, we will apply a regex filter `re.sub(r'^\s*\d{5}\s*', '', text)` to strip all prepended 5-digit DWP paragraph numbers (e.g. `17001`, `18001`) from the `output` policy contexts.
* Delete the existing `cmg_qa_training_data.jsonl` file and re-run the generator script to populate a clean, style-free Q&A dataset of 80 samples.

### 2. Hyperparameter Regularization

#### [MODIFY] [02_train_lora_windows_cpu.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/02_train_lora_windows_cpu.py)
We will adjust LoRA parameters to avoid overfitting and catastrophic forgetting:
* Reduce LoRA Rank from `32` to `16` and Alpha from `64` to `32`. This reduces the parameter capacity of the adapter, preventing it from hard-memorizing the tiny 80-sample dataset.
* Reduce the learning rate from `5e-4` to `1e-4` for gentler, more stable updates.
* Maintain sequence length `max_length=128` and run for exactly 2 epochs (80 steps).
* Keep PyTorch thread counts strictly set to 1 for absolute stability on Windows 11.

### 3. Inference & Decoding Adjustments

#### [MODIFY] [04_evaluate_ab_test.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/04_evaluate_ab_test.py)
We will transition the evaluation decoding strategy from strict greedy search to nucleus sampling to mitigate loops:
* Update both base and tuned generation calls to use:
  ```python
  do_sample=True,
  temperature=0.7,
  top_p=0.9,
  repetition_penalty=1.2,
  max_new_tokens=150
  ```

---

## Verification Plan

### Automated Tests
1. **Regenerate Data:** Delete `cmg_qa_training_data.jsonl` and run `python 01b_ingest_qa_pairs.py` to generate the cleaned dataset. Verify that the file exists and has no prepended paragraph numbers.
2. **Execute Training:** Run `python 02_train_lora_windows_cpu.py` to fine-tune the model with the updated parameters for 80 steps.
3. **Execute Evaluation:** Run `python 04_evaluate_ab_test.py` to output the comparative report.

### Manual Verification
* Inspect the A/B evaluation report to verify:
  1. No prepended codes (e.g. `18001`) appear in the tuned model's output.
  2. Repetitive phrase loops are resolved.
  3. The model maintains a grammatically coherent and professional policy response.
