# DWP CMG RAG Validation Split Evaluation Report

This report compares the performance of the baseline Mistral-7B-Instruct-v0.3, fine-tuned Mistral-7B-Instruct-v0.3, and fine-tuned Qwen-2.5-7B-Instruct models on the full validation split (147 samples).

## Evaluation Metrics Summary

| Model | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Inference Time (s) |
|---|---|---|---|---|
| **Mistral-7B-Instruct (Base + RAG)** | 0.0000 | 0.3434 | 0.8883 | 7.15s |
| **Mistral-7B-Instruct (Tuned + RAG)** | 0.0000 | 0.5052 | 0.9160 | 2.64s |
| **Qwen-2.5-7B-Instruct (Tuned + RAG)** | 0.0000 | 0.4889 | 0.9094 | 2.95s |

## Analysis and Verdict

Based on local semantic evaluation (BERTScore) over all validation samples, the winner is **Mistral-7B-Instruct (Tuned + RAG)**.

- **Mistral-7B-Instruct (Base + RAG)** BERTScore F1: `0.8883`
- **Mistral-7B-Instruct (Tuned + RAG)** BERTScore F1: `0.9160`
- **Qwen-2.5-7B-Instruct (Tuned + RAG)** BERTScore F1: `0.9094`


## Sample Answers Comparison

### Question: *"What happens if the decision is revised under section 16 of the 1991 Act before the appeal is determined?"*

**Ground Truth Context**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal must be treated as though it had been brought against the revised decision. This means that the appeal process continues with the revised decision unless the further representations made by the appellant indicate that the revised decision is not more advantageous to them than the original decision. In such a case, the appeal may lapse.

**Mistral-7B Base Answer**:
> According to Paragraph L_legislation_144, if the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal does not lapse. Instead, the appeal must be treated as though it had been brought against the decision as replaced or revised. The appellant has a period of one month from the date of notification of the decision as replaced or revised to make further representations as to the appeal. After the expiration of this period, or within that period if the appellant consents in writing, the appeal to the First-tier Tribunal must proceed. However, the appeal shall lapse if, in the light of the further representations from the appellant, the decision as replaced or revised is found to be more financially advantageous to the appellant than the decision before it was replaced or revised (Paragraph L_legislation_144, sub-paragraph 1(b)).

Additionally, Paragraph 48018 states that where a decision is revised before the appeal is heard and it is not financially advantageous to the appellant, the original appeal will continue against the newly revised decision. This means that if the revised decision is less favorable to the appellant, the appeal can proceed against the revised decision.

**Mistral-7B Tuned Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will continue against the newly revised decision as per Paragraph L_legislation_144(1)(b) and 48018.

**Qwen-2.5-7B Tuned Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined and it is not more advantageous to the appellant than the original decision, the original appeal will continue against the newly revised decision. This is according to Paragraph 48018 from volume-5-reviews-and-appeals.pdf.

---

### Question: *"What happens if a parentage dispute arises during the MR process?"*

**Ground Truth Context**:
> In exceptional scenarios involving parentage disputes, the normal MR process would still apply as outlined in Paragraph 47008. However, the CMS would take into account the specific nature of parentage disputes and ensure that all relevant parties are involved in the MR process, even if the dispute does not directly involve all parties in every case. The CMS would verify all elements of the decision being reconsidered, not just those raised by the client, and each decision can only go through the MR process once before directing the client to make an appeal to HMCTS.

**Mistral-7B Base Answer**:
> According to the provided policy contexts, if a parentage dispute arises during the Mandatory Reconsideration (MR) process, all parties involved in the case or any other case that could potentially be affected by a change in the decision should be invited to be part of the MR process (Paragraph 47008). The decision being reconsidered will be verified, not just those raised by the client (Paragraph 47008).

However, the policy manual does not provide specific details on how the MR process is conducted in the case of a parentage dispute. For a detailed procedure, it is recommended to refer to Chapter 45: Parentage Disputes, as mentioned in Paragraph 99016 and Paragraph 45002. These sections are not provided in the context, so further research or consultation with a relevant authority may be necessary to find the specific rules for handling parentage disputes during the MR process.

**Mistral-7B Tuned Answer**:
> If a parentage dispute arises during the MR process, the case will be referred to the Parentage Ambassador for advice. The Parentage Ambassador will then liaise with the Child Maintenance Service (CMS) solicitors to determine the appropriate course of action.

**Qwen-2.5-7B Tuned Answer**:
> If a parentage dispute arises during the MR process, the DM should refer to Chapters 45: Parentage Disputes for further guidance on how to proceed. Specifically, DMs should follow the procedures outlined in Chapter 45, which provides detailed steps for handling parentage disputes whether they occur before (pre-MC) or after (post-MC) the initial maintenance calculation has been completed.

---

### Question: *"What should a caseworker do if a parent with care (PWC) contacts them stating they were not notified about the consideration of write-off or the final decision to write off arrears?"*

**Ground Truth Context**:
> The caseworker should record the PWC's concerns and ensure that all communication regarding the consideration of write-off and the final decision to write off arrears have been properly documented and communicated to the PWC. If there has been a failure to notify the PWC, steps should be taken to rectify this by providing the necessary information and ensuring future communications comply with the requirements set out in Paragraph 64032.

**Mistral-7B Base Answer**:
> According to Paragraph 64032 of the provided policy context, if a Parent with Care (PWC) contacts the caseworker stating they were not notified about the consideration of write-off or the final decision to write off arrears, the caseworker may contact the Child Maintenance Service (CMS) to discuss the matter. The caseworker should inform the CMS that they were not notified about the write-off and provide any necessary information to address the situation. The CMS will then review the case and take appropriate action as needed. It is important to note that the policy manual does not provide specific details on the action the CMS should take in response to the caseworker's communication.

**Mistral-7B Tuned Answer**:
> The caseworker should advise the PWC that they have the right to make representations regarding the write-off of arrears. According to Paragraph 64032, the PWC may contact the CMS and express their concerns about not being notified about the write-off process. However, it is important to note that the policy manual does not provide specific guidance on how the CMS should handle this situation beyond advising the PWC of their rights.

**Qwen-2.5-7B Tuned Answer**:
> The caseworker should follow up on the PWC's statement according to Paragraph 64032 from volume-6-collection-and-enforcement-chapters-49-95.pdf. This means the caseworker needs to investigate whether the PWC was indeed not informed about the write-off process and ensure proper communication and notification procedures were followed.

---
