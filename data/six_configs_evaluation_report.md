# DWP CMG Phase 1 Comparative Matrix Evaluation Report

This report compares the performance of 6 distinct configurations across the full validation split (147 gold-standard samples).
The evaluations assess the effects of **Retrieval-Augmented Generation (RAG)** grounding vs. **Parametric Execution (No RAG)**,
as well as **Base Models** vs. **Fine-Tuned Models**.

## Evaluation Metrics Summary

| Configuration | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Latency (s) |
|---|---|---|---|---|
| **Mistral-7B Base + RAG** | 0.0000 | 0.3434 | 0.8883 | 7.14s |
| **Mistral-7B Base (No RAG)** | 0.0000 | 0.1678 | 0.8479 | 9.20s |
| **Mistral-7B Tuned + RAG** | 0.0000 | 0.5052 | 0.9160 | 2.64s |
| **Mistral-7B Tuned (No RAG)** | 0.0000 | 0.3294 | 0.8901 | 2.32s |
| **Qwen-2.5-7B Tuned + RAG** | 0.0000 | 0.4889 | 0.9094 | 2.96s |
| **Qwen-2.5-7B Tuned (No RAG)** | 0.0000 | 0.3202 | 0.8867 | 2.71s |

## Detailed Analysis and Verdict

### Overall Best Configuration: **Mistral-7B Tuned + RAG** (BERTScore F1: `0.9160`)

#### 1. Grounding Impact (RAG vs. No RAG)
- **Average RAG BERTScore**: `0.9046`
- **Average No RAG BERTScore**: `0.8749`
- **Delta**: `+0.0297` in favor of RAG.
Retrieval-Augmentation introduces factual policy context which substantially improves semantic alignment with ground-truth manuals. No-RAG models suffer from hallucinated section titles and paragraph numbers.

#### 2. Domain Fine-Tuning Impact (Tuned vs. Base)
- **Average Tuned BERTScore**: `0.9005`
- **Average Base BERTScore**: `0.8681`
- **Delta**: `+0.0324` in favor of Fine-Tuning.
Fine-tuned models adopt the exact conversational guidelines and structured formatting expected by DWP caseworkers. They respond in a concise, authoritative tone, reducing response verbose and latency (Tuned models average under 3.0s, compared to >7.0s for the base model).

#### 3. Mistral vs. Qwen Fine-Tuned Performance (with RAG)
- **Mistral-7B Tuned + RAG**: `0.9160`
- **Qwen-2.5-7B Tuned + RAG**: `0.9094`
- **Winner**: **Mistral-7B** by a margin of `0.0067`.


## Sample Outputs across all 6 Configurations

### Question: *"What happens if the decision is revised under section 16 of the 1991 Act before the appeal is determined?"*

**Ground Truth (Manual Context)**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal must be treated as though it had been brought against the revised decision. This means that the appeal process continues with the revised decision unless the further representations made by the appellant indicate that the revised decision is not more advantageous to them than the original decision. In such a case, the appeal may lapse.

**Mistral-7B Base + RAG Answer**:
> According to Paragraph L_legislation_144, if the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal does not lapse. Instead, the appeal must be treated as though it had been brought against the decision as replaced or revised. The appellant has a period of one month from the date of notification of the decision as replaced or revised to make further representations as to the appeal. After the expiration of this period, or within that period if the appellant consents in writing, the appeal to the First-tier Tribunal must proceed. However, the appeal shall lapse if, in the light of the further representations from the appellant, the decision as replaced or revised is found to be more financially advantageous to the appellant than the decision before it was replaced or revised (Paragraph L_legislation_144, sub-paragraph 1(b)).

Additionally, Paragraph 48018 states that where a decision is revised before the appeal is heard and it is not financially advantageous to the appellant, the original appeal will continue against the newly revised decision. This means that if the revised decision is less favorable to the appellant, the appeal can proceed against the revised decision.

**Mistral-7B Base (No RAG) Answer**:
> If a decision is revised under Section 16 of the 1991 Act (Asylum and Immigration Act) before the appeal is determined, the appeal process is usually rendered moot, as the original decision being appealed no longer stands.

Section 16 of the 1991 Act allows the Home Office to revoke or vary a decision if it is no longer considered correct or if new information becomes available. If this happens, the appellant may be informed that their appeal has been withdrawn, and they may be required to reapply for asylum or another form of immigration status if applicable.

However, the specific consequences can vary depending on the circumstances of the case, and it is essential to consult with an immigration lawyer or advisor for advice tailored to the individual situation.

**Mistral-7B Tuned + RAG Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will continue against the newly revised decision as per Paragraph L_legislation_144(1)(b) and 48018.

**Mistral-7B Tuned (No RAG) Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will lapse according to section 17 of the same act. This means that the appeal will automatically be cancelled or withdrawn once the decision has been revised.

**Qwen-2.5-7B Tuned + RAG Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined and it is not more advantageous to the appellant than the original decision, the original appeal will continue against the newly revised decision. This is according to Paragraph 48018 from volume-5-reviews-and-appeals.pdf.

**Qwen-2.5-7B Tuned (No RAG) Answer**:
> If the decision is revised under section 16 of the 1991 Act before the appeal is determined, the appeal will lapse according to CS (CE) Regulations 2008, regulation 37(2). This means that the appeal process ceases to exist once the revision is made.

---

### Question: *"What happens if a parentage dispute arises during the MR process?"*

**Ground Truth (Manual Context)**:
> In exceptional scenarios involving parentage disputes, the normal MR process would still apply as outlined in Paragraph 47008. However, the CMS would take into account the specific nature of parentage disputes and ensure that all relevant parties are involved in the MR process, even if the dispute does not directly involve all parties in every case. The CMS would verify all elements of the decision being reconsidered, not just those raised by the client, and each decision can only go through the MR process once before directing the client to make an appeal to HMCTS.

**Mistral-7B Base + RAG Answer**:
> According to the provided policy contexts, if a parentage dispute arises during the Mandatory Reconsideration (MR) process, all parties involved in the case or any other case that could potentially be affected by a change in the decision should be invited to be part of the MR process (Paragraph 47008). The decision being reconsidered will be verified, not just those raised by the client (Paragraph 47008).

However, the policy manual does not provide specific details on how the MR process is conducted in the case of a parentage dispute. For a detailed procedure, it is recommended to refer to Chapter 45: Parentage Disputes, as mentioned in Paragraph 99016 and Paragraph 45002. These sections are not provided in the context, so further research or consultation with a relevant authority may be necessary to find the specific rules for handling parentage disputes during the MR process.

**Mistral-7B Base (No RAG) Answer**:
> In the context of a Medically Assisted Reproduction (MAR) process, such as In-Vitro Fertilization (IVF), a parentage dispute can arise when there is a disagreement about who the legal parents of a child conceived through MAR are. This can happen for various reasons, such as a change in the relationship status of the intended parents, disagreements about the roles and responsibilities of the intended parents, or questions about the biological father.

If a parentage dispute arises during the MAR process, it is important to seek legal advice to understand the specific laws and regulations in your jurisdiction. In general, the following steps may be taken to resolve the dispute:

1. Voluntary Acknowledgment of Parentage: In some cases, the intended parents may be able to resolve the dispute by signing a voluntary acknowledgment of parentage. This document, also known as a "parentage order," establishes the legal parentage of the child and the rights and responsibilities of the parents.
2. Court Order: If the intended parents cannot agree on parentage, a court may need to intervene. The court will consider the best interests of the child and the evidence presented by the parties to determine legal parentage.
3. Paternity Testing: In some cases, paternity testing may be necessary to determine the biological father of the child. This can

**Mistral-7B Tuned + RAG Answer**:
> If a parentage dispute arises during the MR process, the case will be referred to the Parentage Ambassador for advice. The Parentage Ambassador will then liaise with the Child Maintenance Service (CMS) solicitors to determine the appropriate course of action.

**Mistral-7B Tuned (No RAG) Answer**:
> If a parentage dispute arises during the MR process, the MR will lapse and the case will be transferred to the Parentage Ambassador for further investigation. The Parentage Ambassador will then determine the most appropriate course of action based on the circumstances of the case.

**Qwen-2.5-7B Tuned + RAG Answer**:
> If a parentage dispute arises during the MR process, the DM should refer to Chapters 45: Parentage Disputes for further guidance on how to proceed. Specifically, DMs should follow the procedures outlined in Chapter 45, which provides detailed steps for handling parentage disputes whether they occur before (pre-MC) or after (post-MC) the initial maintenance calculation has been completed.

**Qwen-2.5-7B Tuned (No RAG) Answer**:
> If a parentage dispute arises during the maintenance review (MR) process, the decision maker (DM) should refer to Chapter 48: Parentage disputes for further guidance on how to proceed with the case. This chapter provides specific instructions and procedures to address such disputes.

---

### Question: *"What should a caseworker do if a parent with care (PWC) contacts them stating they were not notified about the consideration of write-off or the final decision to write off arrears?"*

**Ground Truth (Manual Context)**:
> The caseworker should record the PWC's concerns and ensure that all communication regarding the consideration of write-off and the final decision to write off arrears have been properly documented and communicated to the PWC. If there has been a failure to notify the PWC, steps should be taken to rectify this by providing the necessary information and ensuring future communications comply with the requirements set out in Paragraph 64032.

**Mistral-7B Base + RAG Answer**:
> According to Paragraph 64032 of the provided policy context, if a Parent with Care (PWC) contacts the caseworker stating they were not notified about the consideration of write-off or the final decision to write off arrears, the caseworker may contact the Child Maintenance Service (CMS) to discuss the matter. The caseworker should inform the CMS that they were not notified about the write-off and provide any necessary information to address the situation. The CMS will then review the case and take appropriate action as needed. It is important to note that the policy manual does not provide specific details on the action the CMS should take in response to the caseworker's communication.

**Mistral-7B Base (No RAG) Answer**:
> In such a situation, the caseworker should follow these steps:

1. Acknowledge the parent's concern: Start by empathizing with the parent and acknowledging their concern about not receiving notification about the consideration or decision of write-off for their arrears.

2. Verify the information: Check the case records to confirm whether the parent was indeed notified as per the usual procedures. This may involve reviewing correspondence, case notes, or contacting the team responsible for sending notifications.

3. Apologize for any oversight: If it is found that the parent was not properly notified, apologize for the oversight and explain that the agency values open communication and regrets any confusion or inconvenience caused.

4. Provide an explanation: Explain the reasons for the write-off, if it has already been decided, and the implications for the parent's ongoing child support obligations. If the decision has not been made yet, inform the parent about the process and timeline for making a decision.

5. Offer a solution: If the parent was not notified, offer to resend the notification or provide a written summary of the information. If the decision has been made, offer to discuss options for resolving the arrears or adjusting the payment plan, if appropriate.

6. Update the case record: Regardless of the outcome, ensure that the case record is updated

**Mistral-7B Tuned + RAG Answer**:
> The caseworker should advise the PWC that they have the right to make representations regarding the write-off of arrears. According to Paragraph 64032, the PWC may contact the CMS and express their concerns about not being notified about the write-off process. However, it is important to note that the policy manual does not provide specific guidance on how the CMS should handle this situation beyond advising the PWC of their rights.

**Mistral-7B Tuned (No RAG) Answer**:
> The caseworker should apologize for the oversight and inform the PWC of their right to appeal the decision within 30 days from the date the letter was sent. Additionally, the caseworker should ensure that all future communications regarding write-off are properly documented and sent to both parties to maintain transparency and comply with the guidelines.

**Qwen-2.5-7B Tuned + RAG Answer**:
> The caseworker should follow up on the PWC's statement according to Paragraph 64032 from volume-6-collection-and-enforcement-chapters-49-95.pdf. This means the caseworker needs to investigate whether the PWC was indeed not informed about the write-off process and ensure proper communication and notification procedures were followed.

**Qwen-2.5-7B Tuned (No RAG) Answer**:
> The caseworker should ensure that the PWC is informed about the consideration of write-off and the final decision to write off arrears as per Paragraph 58014. If there has been a failure to notify the PWC, the caseworker must rectify this oversight and provide the necessary information to the PWC.

---
