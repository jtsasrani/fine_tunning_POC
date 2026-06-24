# DWP CMS Decision Support Suite: RAG & Citation Guide for Decision Makers

This guide provides Decision Makers and Caseworkers with a clear overview of how the RAG (Retrieval-Augmented Generation) search system, citation markers, and relevance scores work in the Decision Support Suite.

---

## 🔍 How Policy Search Works

When you submit a casework question, the AI does not just guess the answer from memory. Instead, the search engine runs a **Hybrid Search** against the indexed policy guides:
1. **Semantic (Vector) Search**: Converts your question into a mathematical fingerprint and finds paragraphs with similar concepts, even if they use different words.
2. **Keyword (Lexical) Search**: Looks for exact terminology matches (e.g., "NRP", "gross weekly income", "variance rule").
3. **Cross-Encoder Reranking**: Takes the combined search results and runs them through a second evaluation step to assign a precise **Rerank Score** based on how accurately the paragraph answers your specific question.

---

## 📊 Understanding the Rerank Score

The Rerank Score is based on a **logit scale** (real numbers ranging from $-\infty$ to $+\infty$), rather than standard percentages or 0-to-1 probabilities.

Here is how to interpret the scores in practice:

| Rerank Score | Relevance Tier | AI Prompt Inclusion | UI Visual Styling |
| :--- | :--- | :--- | :--- |
| **Score $\ge$ 1.0** | **Highly Relevant** | Included in Context | Solid border, cyan badge. Represents a direct, explicit answer to your question. |
| **Score -1.0 to 1.0** | **Semantically Relevant** | Included in Context | Solid border, purple/gray badge. Useful background information or tangential policy context. |
| **Score $<$ -1.0** | **Irrelevant / Low Quality** | **Omitted** (Excluded) | **Dashed border, 60% opacity (dimmed)**, red score badge marked as `(Excluded)`. |

---

## 🚫 Why Are Low-scoring Passages Excluded?

If the AI reads irrelevant or low-quality paragraphs, it can become confused or invent facts (known as **hallucination**). To protect the integrity of the casework:
* Passages with a score **below -1.0** are stripped from the reference material fed to the model before it generates the answer.
* However, the system **still displays them in the citations drawer and RAG Explorer** so you can inspect what the search engine found and double-check the policy context yourself.

---

## 📄 Clicking Badges to Open Documents

In both the Chat Citations drawer and the RAG Explorer panel, the document badges (e.g. `2012-System-Overview.pdf`) are interactive:
* **Action**: Click the document name.
* **Result**: The original policy PDF or HTML guide will open directly in a new browser tab.
* **Inline Reading**: The PDF will render inline inside the browser, allowing you to scroll, search, and verify the paragraph text in its original official layout without downloading the file.

---

## 💡 Tips for Writing Effective Casework Queries

To get the highest retrieval scores and the most accurate AI answers, write your queries using these best practices:

* **Use CMS-Specific Terminology**: Use standard terminology rather than informal phrases.
  * *Instead of*: "what happens if the paying father gets a raise"
  * *Use*: "gross weekly income variation rules for NRP"
* **Provide Contextual Details**: Give details about the specific policy condition.
  * *Instead of*: "pension rules"
  * *Use*: "deducting pension contributions from gross weekly income calculations"
* **Cite Sections if Known**: If you are referencing a specific regulation chapter, mention it.
  * *Instead of*: "enforcement orders"
  * *Use*: "liability order magistrates court process"
