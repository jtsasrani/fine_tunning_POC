# Expert Evaluation: AWS Migration Plan — Go/No-Go Assessment

> **Verdict: ✅ GREEN LIGHT — Proceed with confidence.**
> The plan is well-structured, technically sound, and financially conservative. Below I detail what's excellent, what needs minor updates, and my recommendation on model size.

---

## 1. Overall Plan Quality: 9/10

| Dimension | Rating | Notes |
|---|---|---|
| **Technical Accuracy** | ⭐⭐⭐⭐⭐ | QLoRA configs, VRAM estimates, instance selection — all correct |
| **Budget Planning** | ⭐⭐⭐⭐⭐ | $2,500 is genuinely generous; cost estimates are conservative (good) |
| **Risk Mitigation** | ⭐⭐⭐⭐ | Spot resilience via S3 sync is solid; could add one more safeguard (see below) |
| **Phasing & Dependencies** | ⭐⭐⭐⭐⭐ | Logical ordering, correct parallelization (Phase 2 ∥ Phase 3) |
| **Legal/Compliance** | ⭐⭐⭐⭐⭐ | Excellent DWP-specific analysis including Data (Use and Access) Act 2025 |
| **Tooling (2026 Currency)** | ⭐⭐⭐ | Missing Unsloth framework; one new model contender to consider (see §3) |

### What's Excellent
- **Region choice** (`eu-west-2`): Mandatory for UK government data residency. Correct call.
- **Spot instance strategy**: At ~$0.55/hr you get ~4,400 hours — far more than you'll ever need.
- **QLoRA configuration**: The config is production-grade (NF4, double quant, paged AdamW, gradient checkpointing). No changes needed.
- **Data pipeline strategy**: Using an external high-quality LLM (Claude/GPT-4o-mini) to generate training data from your existing 1,779 paragraphs is the right approach. It breaks the "garbage-in, garbage-out" cycle.
- **Two-stage retrieval** (BM25+Vector → Cross-Encoder reranking): Industry best practice. Excellent upgrade.
- **Budget buffer** (35%): Smart. Most projects don't plan for this.

---

## 2. Phase-by-Phase Review

### Phase 1: Infrastructure ✅ No Issues

| Item | Status | Comment |
|---|---|---|
| g5.xlarge (A10G, 24GB VRAM) | ✅ Perfect | Right GPU for 7B–14B QLoRA |
| Deep Learning AMI | ✅ Correct | Pre-installed CUDA + PyTorch saves hours of setup |
| 100GB gp3 SSD | ✅ Adequate | 7B model weights ~14GB + data + checkpoints = ~40-50GB |
| S3 persistence | ✅ Essential | Critical for spot instance resilience |
| VS Code Remote SSH | ✅ Seamless | Zero workflow disruption — exactly what you need |

> [!TIP]
> **Minor addition**: Set up an **AWS Budget Alert** at $500, $1,000, $1,500, and $2,000. This is a 5-minute task in the AWS console that prevents surprise charges.

### Phase 2: Model Upgrade ✅ Sound, With Updates (See §3 below)

The core approach (360M → 7B via QLoRA) is the single highest-impact change. The training config is correct. My detailed model recommendation is in Section 3.

### Phase 3: Data Pipeline ✅ Excellent Design

| Strategy | Assessment |
|---|---|
| **A: LLM API generation** (~1,500 samples) | ✅ Best strategy. $5-15 cost is negligible. |
| **B: Public data expansion** (~500-1,000) | ✅ Smart long-term move. GOV.UK data is public and freely usable. |
| **C: Augmentation** (~500) | ⚠️ Good but lowest priority. Focus on A and B first. |

> [!IMPORTANT]
> **Data quality > Data quantity**. With a 7B model, 1,000 high-quality samples will outperform 3,000 mediocre ones. I recommend:
> 1. Start with Strategy A (LLM-generated from existing PDFs) — this alone may be sufficient
> 2. Evaluate model performance after training on just Strategy A data
> 3. Only add B and C if evaluation shows a need

### Phase 4: RAG Upgrades ✅ All Correct

| Upgrade | Assessment |
|---|---|
| `bge-base-en-v1.5` embeddings | ✅ Significant upgrade from MiniLM. Consider also `nomic-embed-text-v1.5` |
| FAISS vector store | ✅ Industry standard. Overkill for 1,779 chunks but future-proofs |
| Cross-encoder reranking | ✅ Best practice for precision |
| Improved chunking (500 tokens, 100 overlap) | ✅ Much better than raw paragraph splitting |

### Phase 5: Evaluation ✅ Well Designed

The 5-metric evaluation framework (ROUGE-L, BERTScore, Factual Accuracy, Hallucination Rate, Retrieval Precision) is thorough.

> [!TIP]
> **Add one more metric**: Use **LLM-as-a-Judge** — have GPT-4 or Claude score your model's answers on a 1-5 scale for accuracy, completeness, and helpfulness. This correlates more closely with human judgment than automated metrics.

### Phase 6: Deployment ✅ Sensible

vLLM is the right choice for inference optimization. 3-6× speedup is realistic on A10G.

> [!NOTE]
> For the initial proof-of-concept demo, you can skip vLLM entirely and just use HuggingFace `generate()`. vLLM is a production optimization — not needed for the first demo to stakeholders.

---

## 3. Model Selection: The Big Question

### Your Inclination: Mistral-7B-Instruct-v0.3

**This is a solid, defensible choice.** Here's my honest assessment:

| Factor | Mistral-7B Assessment |
|---|---|
| **License** | Apache 2.0 — **zero legal friction** for DWP. No agreements, no branding requirements. This matters for government. |
| **Instruction Following** | Excellent. Known for consistent, predictable output formatting — critical for a decision-maker tool. |
| **Fine-tuning Ecosystem** | Mature. Widely supported in every framework (Unsloth, Axolotl, PEFT). |
| **Inference Speed** | Fast. GQA + Sliding Window Attention make it one of the most efficient 7B models. |
| **RAG Grounding** | Good. Follows retrieved context faithfully when properly prompted. |
| **Reasoning Depth** | Good but not best-in-class (Qwen3-8B is stronger here). |

### Should You Consider a Bigger Model (13B–14B)?

**My honest recommendation: Start with 7B–8B. Do NOT jump to 14B yet.**

Here's why:

| Argument | For 14B | Against 14B |
|---|---|---|
| **Quality** | Better reasoning, less hallucination | Marginal for your use case (RAG provides the knowledge, not the model) |
| **VRAM** | Fits on A10G with QLoRA (~12-14GB) | Tighter — smaller batch sizes, less room for experimentation |
| **Training Speed** | — | ~2× slower per run (fewer experiments per dollar) |
| **Inference** | — | Slower inference = worse user experience |
| **Your Budget** | Can afford it | Halves your experimentation capacity |

> [!IMPORTANT]
> **The key insight**: Your system is RAG-based. The model doesn't need to *memorize* policy knowledge — it needs to *reason over retrieved context*. A well-fine-tuned 7B model with excellent RAG is better than a mediocre 14B model with poor RAG. Your bottleneck is **retrieval quality and training data quality**, not model size.

**If evaluation later shows the 7B is struggling with complex multi-paragraph reasoning**, THEN upgrade to 14B. Your A10G supports it. But start small, iterate fast.

### Updated Model Shortlist for 2026

The plan lists Mistral-7B, Llama-3.1-8B, and Phi-3. Since this plan was written, **one strong new contender has emerged**:

| Model | Params | Context | License | Key Advantage | Consideration |
|---|---|---|---|---|---|
| **Mistral-7B-Instruct-v0.3** | 7.2B | 32K | Apache 2.0 ✅ | Best consistency & legal simplicity | Your current top pick |
| **Qwen3-8B** ⭐ NEW | 8B | 32K | Apache 2.0 ✅ | Strongest reasoning in class; "thinking" mode | Newer, fewer tutorials but excellent benchmarks |
| **Llama-3.1-8B-Instruct** | 8B | 128K | Custom | Largest ecosystem | Requires license acceptance + "Built with Llama" branding |
| **Phi-3-small (8k)** | 7.4B | 8K | MIT ✅ | Very efficient | 8K context is limiting — policy docs can be long |

### My Final Model Recommendation

```
Primary:   Mistral-7B-Instruct-v0.3  ← Your inclination is correct
Benchmark: Qwen3-8B                  ← Add this to your benchmark (same cost to test)
Skip:      Phi-3 (8K context too short for policy docs)
Optional:  Llama-3.1-8B (good but has legal friction for government)
```

**Mistral-7B is the right starting point** because:
1. ✅ Apache 2.0 = zero legal review needed for DWP
2. ✅ Known for consistent output formatting (critical for a decision-maker tool)
3. ✅ Fastest inference in the 7B class
4. ✅ Most mature fine-tuning ecosystem
5. ✅ 32K context window is more than sufficient

**But also benchmark Qwen3-8B** because:
1. It leads 2026 benchmarks in reasoning tasks
2. Same Apache 2.0 license — no additional legal burden
3. Testing both costs under $5 total

---

## 4. Critical Update: Use Unsloth for Fine-Tuning

The current plan uses standard HuggingFace `transformers` + `peft`. **In 2026, Unsloth has become the industry standard** for single-GPU QLoRA training.

| Metric | Standard HF + PEFT | Unsloth |
|---|---|---|
| Training Speed | Baseline | **~2× faster** |
| VRAM Usage | ~10-12 GB (7B QLoRA) | **~5-7 GB (70% less)** |
| Context Length | Limited by VRAM | **Up to 13× longer** |
| Accuracy | Baseline | **Identical** (no lossy shortcuts) |
| Setup Complexity | Moderate | **Simple** (drop-in replacement) |

> [!IMPORTANT]
> **Action**: Replace the training pipeline's use of raw `transformers` + `peft` with **Unsloth**. This is a code-level change in `02_train_qlora_gpu.py` — the config values (LoRA rank, learning rate, etc.) stay the same. The benefit is enormous: 2× faster training means 2× more experiments within your budget.

### What Unsloth Enables
- Train 7B model in **~45 minutes** instead of ~1.5 hours (for 500 samples)
- Frees enough VRAM to **increase max_length to 1024** (from 512) — better policy coverage
- Makes **14B models easily feasible** if you later decide to upgrade

---

## 5. Budget Validation ✅

| Scenario | Estimated Cost | Within Budget? |
|---|---|---|
| Phase 1 (infrastructure + setup) | ~$5-10 | ✅ |
| Phase 2 (3 model benchmarks × 3 epochs) | ~$6-10 | ✅ |
| Phase 3 (LLM API for data generation) | ~$15-20 | ✅ |
| Phase 4 (RAG rebuild — compute time) | ~$2-5 | ✅ |
| Phase 5 (evaluation runs) | ~$5-10 | ✅ |
| Phase 6 (deployment + testing) | ~$50-100 | ✅ |
| **Total estimated core cost** | **~$100-155** | ✅ |
| **Remaining for experimentation** | **~$2,345-2,400** | 💰 |

> [!NOTE]
> You will spend roughly **$100-150 to complete the entire migration and build a production-grade system**. The remaining ~$2,350 gives you an enormous runway for experimentation, hyperparameter tuning, and scaling.

---

## 6. Risk Assessment Update

| Risk | Severity | Mitigation in Plan | Additional Recommendation |
|---|---|---|---|
| Spot instance interruption | Medium | ✅ S3 sync + checkpoints | Add: **Use `--save_steps=100`** in training args for more frequent checkpoints |
| Model hallucination persists | Medium | ✅ Evaluation framework | Add: **LLM-as-Judge** evaluation alongside automated metrics |
| Training data quality | Medium | ✅ Multi-source approach | Add: **Human review of 50 random samples** before first training run |
| `eu-west-2` spot availability | Low | ⚠️ Not addressed | Add: **Set fallback to `eu-west-1` (Ireland)** — still EU, still UK-adjacent |
| Overfitting on small dataset | Low | Partially addressed | Add: **Use validation loss monitoring** with early stopping |

---

## 7. Recommended Execution Order

Here's my suggested priority order when you start:

### Week 1: Foundation
- [ ] **Phase 1**: Launch EC2, connect VS Code, set up S3
- [ ] **Phase 3A**: Generate training data using LLM API (can run overnight)
- [ ] Install Unsloth on EC2

### Week 2: Model Training & Evaluation
- [ ] **Phase 2**: Fine-tune Mistral-7B with Unsloth + QLoRA
- [ ] **Phase 2+**: Benchmark Qwen3-8B on same data (costs $2)
- [ ] **Phase 5**: Run evaluation framework, pick winner

### Week 3: RAG & Deployment
- [ ] **Phase 4**: Upgrade embeddings, FAISS, cross-encoder reranking
- [ ] **Phase 6**: Deploy FastAPI endpoint
- [ ] Demo to stakeholders

---

## 8. Summary: Go/No-Go Checklist

| Checkpoint | Status |
|---|---|
| Is the plan technically sound? | ✅ Yes |
| Is the budget sufficient? | ✅ Yes — by a factor of 15-20× |
| Is the model choice correct? | ✅ Yes — Mistral-7B is the right starting point |
| Should you use a bigger model? | ❌ Not yet — start 7B, upgrade later if needed |
| Are there legal/compliance blockers? | ✅ No — Apache 2.0 + eu-west-2 + self-hosted |
| Is the tooling current? | ⚠️ Update: Add Unsloth framework |
| Are risks adequately mitigated? | ✅ Yes, with minor additions above |
| **FINAL VERDICT** | **✅ PROCEED** |

> [!IMPORTANT]
> **Bottom line**: You have a well-architected plan, a generous budget, and a clear path forward. Your instinct on Mistral-7B is correct. The only material update I recommend is adopting **Unsloth** for training (2× speed, 70% less VRAM) and adding **Qwen3-8B** to the benchmark list. Everything else in the plan is solid. Go build it. 🚀
