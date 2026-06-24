# Implementation Plan - vLLM Backend Migration

This plan details the migration of the DWP CMS Decision Support Suite backend inference engine from standard PyTorch/Transformers (`AutoModelForCausalLM` with bitsandbytes) to **vLLM** (`vllm.LLM`). This change aims to optimize generation latency from 6–7 seconds down to **1–2 seconds** while enabling high-throughput concurrent query serving.

## Proposed Architecture
```mermaid
graph LR
    User[Caseworker Query] --> WebUI[Web Interface]
    WebUI --> API[/api/query/generate]
    API --> vLLM[vLLM Engine]
    vLLM --> VRAM[4-Bit quantized GPU weights]
    VRAM --> Response[Optimized Response <2s]
```

---

## User Review Required

> [!IMPORTANT]
> **Library Installation Required**: We will install `vllm` (version 0.23.0+) and upgrade `torch` and `transformers` dependencies on the remote GPU instance. The pip installer dry-run completed successfully with zero dependency resolution conflicts.

> [!WARNING]
> **Startup Time Trade-off**: The initial model loading time on backend startup might increase by ~30–40 seconds because vLLM pre-allocates KV-caches for PagedAttention in VRAM. However, subsequent chat response generation will drop significantly (under 2 seconds per query).

---

## Open Questions

> [!NOTE]
> We have confirmed that `vllm` supports Python 3.13 and the bitsandbytes 4-bit quantization format natively. No new quantizations of the merged model weights are necessary.

---

## Proposed Changes

### Backend Components

#### [MODIFY] [app.py](file:///home/ubuntu/dwp-cmg-finetune/app.py)
* **vLLM Engine Initialization**: Modify `initialize_models()` to import `vllm.LLM` and instantiate the engine targeting the merged LoRA model directory.
* **Quantization Configuration**: Load the model using `quantization="bitsandbytes"` to ensure it respects the 24GB VRAM limit of the A10G GPU.
* **Generate Endpoint Update**: Rewrite the inference generation block in `/api/query/generate` to use `llm_engine.generate` with `vllm.SamplingParams` instead of standard `model.generate()`.

---

## Detailed Code Modifications (Draft)

### app.py Loader Changes:
```python
# Global reference for vLLM engine
llm_engine = None

def initialize_models():
    global db_data, faiss_index, embed_tokenizer, embed_model, reranker, tokenizer_dict, llm_engine, DEMO_MODE
    
    # ... (FAISS / Embeddings / Cross-Encoder load as usual) ...
    
    primary_model_path = "./trained_models/qwen-14b-cms-qlora_merged"
    if not os.path.exists(primary_model_path):
        print(f"Warning: Tuned model path '{primary_model_path}' not found. Auto-enabling Demo Mode.")
        DEMO_MODE = True
        
    if DEMO_MODE:
        print("=== RUNNING IN DEMO MODE ===")
        return
        
    # Initialize vLLM Engine
    from vllm import LLM
    print("Configuring and loading vLLM engine for Qwen-14B CMS...", flush=True)
    
    # Load tokenizer for chat template parsing
    tokenizer_dict["qwen_14b_tuned"] = AutoTokenizer.from_pretrained(primary_model_path)
    
    # Initialize the LLM engine in 4-bit quantization
    llm_engine = LLM(
        model=primary_model_path,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.90,
        max_model_len=4096
    )
    print("vLLM engine loaded successfully!", flush=True)
```

### app.py Generator Changes:
```python
        # Build prompt using tokenizer's template
        prompt = tok.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Configure generation parameters via vLLM SamplingParams
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            repetition_penalty=1.2,
            stop_token_ids=[tok.eos_token_id]
        )
        
        t_start = time.time()
        # Run vLLM generation
        outputs = llm_engine.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        elapsed_time = time.time() - t_start
```

---

## Verification Plan

### Automated Verification
* Run pip installation command on the remote server:
  `/opt/pytorch/bin/pip install vllm`
* Execute verification check to test imports and versioning:
  `/opt/pytorch/bin/python3 -c "import vllm; print('vLLM loaded successfully')"`

### Manual Verification
* Start the Flask server on the remote instance in production mode.
* Monitor `data/flask_server.log` to verify successful engine pre-allocation and weight loading.
* Use `nvidia-smi` to monitor GPU memory allocation (should hover around ~18-20 GB VRAM).
* Visit `http://127.0.0.1:5000` locally, send multiple test queries to the assistant, and verify that responses generate successfully in under 2 seconds.
