# DWP CMG RAG Model Evaluation Report

This report compares the performance of the fine-tuned Mistral-7B-Instruct-v0.3 and Qwen-2.5-7B-Instruct models on the 30 gold-standard questions.

## Evaluation Metrics Summary

| Model | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Inference Time (s) |
|---|---|---|---|---|
| **Mistral-7B-Instruct** | 0.0000 | 0.5517 | 0.9248 | 2.53s |
| **Qwen-7B-Instruct** | 0.0000 | 0.5083 | 0.9117 | 2.75s |

## Analysis and Verdict

Based on local semantic evaluation (BERTScore), the winner is **Mistral-7B**.
- Mistral-7B BERTScore: `0.9248`
- Qwen-2.5-7B BERTScore: `0.9117`


## Sample Answers Comparison

### Question: *"Should the DM identify the case as not being suitable for manual representation and write off, what should they do next?"*

**Ground Truth Context**:
> In such cases, the DM should return the case to Business as Usual (BAU) for collection and enforcement, as per Chapter 54: Enforcement and Collection of CSA debt on CMS.

**Mistral-7B Answer**:
> The DM should return the case to BAU for collection and enforcement, referring to Chapter 54: Enforcement and Collection of CSA debt on CMS for further details.

**Qwen-2.5-7B Answer**:
> If the DM identifies the case as not being suitable for manual representation and write off, the case should be returned to BAU for collection and enforcement. This is according to Paragraph 64006 from volume-6-collection-and-enforcement-chapters-49-95.pdf.

---

### Question: *"When should sanctions be applied according to Paragraph 90007?"*

**Ground Truth Context**:
> Sanctions should only be applied after considering all other Scottish enforcement actions for each LO period. This means that before applying sanctions, all available Scottish enforcement methods must be exhausted first.

**Mistral-7B Answer**:
> Sanctions should be applied after all other Scottish enforcement actions have been considered on each LO period as per Paragraph 90007.

**Qwen-2.5-7B Answer**:
> According to Paragraph 90007, all other Scottish enforcement actions must be considered before applying for sanctions on all cases for each LO period. This refers to Chapter 81: Legal Enforcement for further details.

---

### Question: *"If multiple applicants have parental responsibility or rights, how does the DWP determine which application takes priority?"*

**Ground Truth Context**:
> According to Paragraph 37020, if more than one applicant has parental responsibility or rights, the general rule will apply and the earliest application will take priority.

**Mistral-7B Answer**:
> The DWP determines which application takes priority by applying the general rule that the earliest application will take precedence. However, it should be noted that this determination requires evidence to confirm which applicant has parental responsibility or rights. Refer to Paragraphs 37020, 99014, and 05059 for further details.

**Qwen-2.5-7B Answer**:
> The DWP determines which application takes priority by applying the general rule and considering the earliest application date. However, first, DMs must use available evidence to establish who among the applicants has parental responsibility for the Qualified Child (QC). This process is outlined in Paragraph 99014 from Volume 7 - Evidence and Decision Making. For further guidance on handling multiple applications, refer to Chapter 37: Applications.

---
