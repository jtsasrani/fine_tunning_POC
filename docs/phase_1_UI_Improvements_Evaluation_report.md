# DWP CMG Phase 1 Comparative Matrix Evaluation Report

This report compares the performance of 6 distinct configurations across the full validation split (147 gold-standard samples).
The evaluations assess the effects of **Retrieval-Augmented Generation (RAG)** grounding vs. **Parametric Execution (No RAG)**,
as well as **Base Models** vs. **Fine-Tuned Models**.

## Evaluation Metrics Summary

| Configuration | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Latency (s) |
|---|---|---|---|---|
| **Mistral-7B Base + RAG** | 0.0000 | 0.3434 | 0.8883 | 7.14s |
| **Mistral-7B Base (No RAG)** | 0.0000 | 0.1678 | 0.8479 | 9.20s |
| **Mistral-7B Tuned + RAG** | 0.0000 | 0.5052 | **0.9160** | **2.64s** |
| **Mistral-7B Tuned (No RAG)** | 0.0000 | 0.3294 | 0.8901 | 2.32s |
| **Qwen-2.5-7B Tuned + RAG** | 0.0000 | 0.4889 | 0.9094 | 2.96s |
| **Qwen-2.5-7B Tuned (No RAG)** | 0.0000 | 0.3202 | 0.8867 | 2.71s |

---

## Detailed Analysis and Verdict

### Overall Best Configuration: **Mistral-7B Tuned + RAG** (BERTScore F1: `0.9160`)

#### 1. Grounding Impact (RAG vs. No RAG)
*   **Average RAG BERTScore**: `0.9046`
*   **Average No RAG BERTScore**: `0.8749`
*   **Delta**: `+0.0297` in favor of RAG.
*   **Insight**: Retrieval-Augmentation introduces factual policy context which substantially improves semantic alignment with ground-truth manuals. Without RAG, models tend to hallucinate general rules, non-existent acts, or guess numbers. For example, without RAG, the base model guesses that a parentage dispute in a child maintenance context is related to IVF or medical birth-parent registries.

#### 2. Domain Fine-Tuning Impact (Tuned vs. Base)
*   **Average Tuned BERTScore**: `0.9005`
*   **Average Base BERTScore**: `0.8681`
*   **Delta**: `+0.0324` in favor of Fine-Tuning.
*   **Insight**: Fine-tuned models adopt the exact conversational guidelines and structured formatting expected by DWP caseworkers. They respond in a concise, authoritative tone, reducing verbose outputs and latency (Tuned models average under 3.0s, compared to >7.0s for the base model).

#### 3. Mistral vs. Qwen Fine-Tuned Performance (with RAG)
*   **Mistral-7B Tuned + RAG**: `0.9160`
*   **Qwen-2.5-7B Tuned + RAG**: `0.9094`
*   **Winner**: **Mistral-7B** by a margin of `0.0067`.

---

## Sample Outputs across all 6 Configurations

### Question: *"What happens if the decision is revised under section 16 of the 1991 Act before the appeal is determined?"*

*   **Ground Truth (Manual Context)**:
    > If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal must be treated as though it had been brought against the revised decision. This means that the appeal process continues with the revised decision unless the further representations made by the appellant indicate that the revised decision is not more advantageous to them than the original decision. In such a case, the appeal may lapse.
*   **Mistral-7B Base + RAG Answer**:
    > According to Paragraph L_legislation_144, if the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal does not lapse. Instead, the appeal must be treated as though it had been brought against the decision as replaced or revised...
*   **Mistral-7B Base (No RAG) Answer**:
    > If a decision is revised under Section 16 of the 1991 Act (Asylum and Immigration Act) before the appeal is determined, the appeal process is usually rendered moot... (Hallucination)
*   **Mistral-7B Tuned + RAG Answer**:
    > If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will continue against the newly revised decision as per Paragraph L_legislation_144(1)(b) and 48018. (Highly concise and accurate)
*   **Mistral-7B Tuned (No RAG) Answer**:
    > If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will lapse according to section 17 of the same act.
*   **Qwen-2.5-7B Tuned + RAG Answer**:
    > If the decision is revised under section 16 of the 1991 Act before the appeal is determined and it is not more advantageous to the appellant than the original decision, the original appeal will continue against the newly revised decision. This is according to Paragraph 48018 from volume-5-reviews-and-appeals.pdf.
*   **Qwen-2.5-7B Tuned (No RAG) Answer**:
    > If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will lapse according to CS (CE) Regulations 2008, regulation 37(2).
