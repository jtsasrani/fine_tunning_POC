# DWP CMG RAG Validation Split Evaluation Report

This report compares the performance of the fine-tuned Mistral-7B-Instruct-v0.3 and Qwen-2.5-7B-Instruct models on the full validation split (147 samples).

## Evaluation Metrics Summary

| Model | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Inference Time (s) |
|---|---|---|---|---|
| **Mistral-7B-Instruct (Tuned)** | 0.0000 | 0.5052 | 0.9160 | 2.64s |
| **Qwen-7B-Instruct (Tuned)** | 0.0000 | 0.4889 | 0.9094 | 2.96s |

## Analysis and Verdict

Based on local semantic evaluation (BERTScore) over all validation samples, the winner is **Mistral-7B**.
- Mistral-7B BERTScore: `0.9160`
- Qwen-2.5-7B BERTScore: `0.9094`


## Sample Answers Comparison

### Question: *"What happens if the decision is revised under section 16 of the 1991 Act before the appeal is determined?"*

**Ground Truth Context**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal must be treated as though it had been brought against the revised decision. This means that the appeal process continues with the revised decision unless the further representations made by the appellant indicate that the revised decision is not more advantageous to them than the original decision. In such a case, the appeal may lapse.

**Mistral-7B Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will continue against the newly revised decision as per Paragraph L_legislation_144(1)(b) and 48018.

**Qwen-2.5-7B Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined and it is not more advantageous to the appellant than the original decision, the original appeal will continue against the newly revised decision. This is according to Paragraph 48018 from volume-5-reviews-and-appeals.pdf.

---

### Question: *"What happens if a parentage dispute arises during the MR process?"*

**Ground Truth Context**:
> In exceptional scenarios involving parentage disputes, the normal MR process would still apply as outlined in Paragraph 47008. However, the CMS would take into account the specific nature of parentage disputes and ensure that all relevant parties are involved in the MR process, even if the dispute does not directly involve all parties in every case. The CMS would verify all elements of the decision being reconsidered, not just those raised by the client, and each decision can only go through the MR process once before directing the client to make an appeal to HMCTS.

**Mistral-7B Answer**:
> If a parentage dispute arises during the MR process, the case will be referred to the Parentage Ambassador for advice. The Parentage Ambassador will then liaise with the Child Maintenance Service (CMS) solicitors to determine the appropriate course of action.

**Qwen-2.5-7B Answer**:
> If a parentage dispute arises during the MR process, the DM should refer to Chapters 45: Parentage Disputes for further guidance on how to proceed. Specifically, DMs should follow the procedures outlined in Chapter 45, which provides detailed steps for handling parentage disputes whether they occur before (pre-MC) or after (post-MC) the initial maintenance calculation has been completed.

---

### Question: *"What should a caseworker do if a parent with care (PWC) contacts them stating they were not notified about the consideration of write-off or the final decision to write off arrears?"*

**Ground Truth Context**:
> The caseworker should record the PWC's concerns and ensure that all communication regarding the consideration of write-off and the final decision to write off arrears have been properly documented and communicated to the PWC. If there has been a failure to notify the PWC, steps should be taken to rectify this by providing the necessary information and ensuring future communications comply with the requirements set out in Paragraph 64032.

**Mistral-7B Answer**:
> The caseworker should advise the PWC that they have the right to make representations regarding the write-off of arrears. According to Paragraph 64032, the PWC may contact the CMS and express their concerns about not being notified about the write-off process. However, it is important to note that the policy manual does not provide specific guidance on how the CMS should handle this situation beyond advising the PWC of their rights.

**Qwen-2.5-7B Answer**:
> The caseworker should follow up on the PWC's statement according to Paragraph 64032 from volume-6-collection-and-enforcement-chapters-49-95.pdf. This means the caseworker needs to investigate whether the PWC was indeed not informed about the write-off process and ensure proper communication and notification procedures were followed.

---
