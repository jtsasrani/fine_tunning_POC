# User Guide: RAG & Citation Interpretation for Decision Makers

**Document Control**
* **Title**: DWP CMS Decision Support Suite - Caseworker Search & Citation Guide
* **Version**: 2.0 (Production Release)
* **Author**: Operations Product Owner
* **Target Audience**: CMS Caseworkers, Decision Makers, Appeals Officers
* **Status**: Approved

---

## 🔍 1. Behind the Scenes: How Policy Search Works

When you submit a query to the AI Decision Support Suite, the system does not generate an answer from raw memory. It executes a multi-stage **Hybrid Search** against the indexed DWP Child Maintenance Service (CMS) manuals:

```
                  ┌──────────────────────────────┐
                  │   Caseworker Search Query    │
                  └──────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Dense Semantic Search           Sparse Keyword Search
       (BGE-base-en-v1.5)              (Lexical Overlap + Term Mapping)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                       Candidate Passage Union
                                 │
                                 ▼
                     Cross-Encoder Reranking
                 (ms-marco-MiniLM-L-6-v2 model)
                                 │
                                 ▼
                      Passage Sorting & Filtering
                 (Threshold: -1.0 Logit Relevance)
```

### Automatic Term Mapping & Query Expansion
To ensure that search results are resilient to different wording, the keyword engine automatically maps common caseworker phrases to their official statutory terms:
* **"paying parent"** or **"parent"** $\to$ automatically matches and indexes against **"NRP"** (Non-Resident Parent).
* **"variance"** $\to$ automatically expands to match **"differ"**, **"differs"**, **"difference"**, **"change"**, or **"changed"**.

This guarantees that searching for a colloquial caseworker scenario (e.g., "what if the paying parent gets unearned income") successfully retrieves official policy paragraphs written using strict legal vocabulary (e.g., "unearned income variance rules for the NRP").

---

## 📊 2. Understanding the Rerank Score

Candidate paragraphs are evaluated by a secondary machine learning model called a **Cross-Encoder Reranker**. This model assigns a score based on how accurately the text answers your question.

### The Logit Scale
Unlike standard percentages ($0\%$ to $100\%$) or scores restricted between $0.0$ and $1.0$, the Reranker uses a **logit scale** (real numbers ranging from negative to positive infinity).

| Rerank Score | Relevance Tier | Prompt Inclusion | UI Visual Styling |
| :--- | :--- | :--- | :--- |
| **Score $\ge$ 1.0** | **Highly Relevant** | Included in Context | **Solid cyan border, cyan badge**. Directly answers the query. |
| **Score -1.0 to 1.0** | **Semantically Relevant** | Included in Context | **Solid purple border, purple/gray badge**. Useful context. |
| **Score $<$ -1.0** | **Low Relevance** | **Omitted from Prompt** | **Dashed red border, 60% opacity (dimmed)**, red score badge labeled `(Excluded)`. |

---

## 🚫 3. Excluded Passages: Why Are They Visible?

If the AI model reads irrelevant paragraphs, it can become confused and generate incorrect or invented answers (known as **hallucinations**). To protect casework integrity:
* Passages with a score **below -1.0** are stripped from the reference material fed to the model before it generates the answer.
* **Why show them?**: The system displays these excluded passages in the **Citations Drawer** and the **RAG Explorer** with a dimmed 60% opacity and dashed border. This allows you to audit the search engine's performance, inspect low-scoring documents, and verify if a policy was correctly excluded.

---

## 📄 4. Interactive PDF Badges & Inline Reading

In the Chat Citations drawer and the RAG Explorer panel, document badges (e.g., `volume-3-variations-chapters-27-36.pdf`) are interactive:

* **Click to Open**: Click any document badge to open the original DWP manual in a new browser tab.
* **Secure Inline Rendering**: The application streams the file directly from the secure server storage `/home/ubuntu/dwp-cmg-finetune/source_pdfs/` to render inline in the browser. You can view the document's original official layout, tables, and surrounding paragraphs without downloading files to your computer.

---

## 💡 5. Tips for Writing Effective Queries

Follow these search guidelines to obtain the highest relevance scores and the most accurate AI answers:

* **Specify the Role or Party**: Use specific terms like "NRP" or "Receiving Parent" if the policy differs by role.
  * *Instead of*: "what happens if a parent gets a pension"
  * *Use*: "how pension contributions affect gross weekly income calculations for the NRP"
* **Target Specific Policy Concepts**: Use official DWP phrasing.
  * *Instead of*: "father has a side job making money"
  * *Use*: "unearned income variance rules"
* **Focus on Policy, Not Action Commands**: Frame queries as statements of inquiry.
  * *Instead of*: "calculate child support for income 500"
  * *Use*: "gross weekly income calculation thresholds and rates"

