# DWP CMG AWS Phase 1 UI Expansion — Implementation Plan

This plan details the implementation of an expanded, 6-card comparison matrix in the DWP CMG decision-support web interface. The expanded matrix will compare **Mistral Base, Mistral Tuned, and Qwen Tuned** under both **Grounded (With RAG)** and **Parametric (No RAG)** conditions.

To manage user wait times and maintain engagement, the client-side JavaScript will execute the model calls **sequentially one-by-one**, rendering each output as soon as it completes.

---

## Proposed Changes

We will edit the backend server, the HTML structure, and the JavaScript client to implement the new configurations.

### 1. Backend API: [app.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/app.py)
We will modify the `handle_generate` endpoint to support closed-book (No RAG) inference:
- Accept a boolean `use_rag` parameter from the JSON request.
- If `use_rag` is `True`, compile the system prompt and retrieved contexts (standard RAG template).
- If `use_rag` is `False`, compile a simple chat prompt: `messages = [{"role": "user", "content": query}]`.

### 2. UI Layout: [templates/index.html](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/templates/index.html)
We will expand the matrix grid from 3 cards to 6 cards, arranged in a **3x2 grid** where each column represents a model and each row represents the RAG status:
- **Row 1 (With RAG - Grounded)**:
  - Card C: Mistral-7B Base + RAG
  - Card A: Mistral-7B Tuned + RAG
  - Card B: Qwen-2.5-7B Tuned + RAG
- **Row 2 (No RAG - Parametric)**:
  - Card D: Mistral-7B Base (No RAG)
  - Card E: Mistral-7B Tuned (No RAG)
  - Card F: Qwen-2.5-7B Tuned (No RAG)

### 3. JavaScript Controller: [static/script.js](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/static/script.js)
We will update the client logic to coordinate sequential fetching:
- Extend the models tracking objects (`skeletons`, `contents`, `times`) to include the keys `a, b, c, d, e, f`.
- Configure the sequence array `modelsToRun` to run the 6 configurations sequentially:
  1. Mistral Base + RAG
  2. Mistral Base (No RAG)
  3. Mistral Tuned + RAG
  4. Mistral Tuned (No RAG)
  5. Qwen Tuned + RAG
  6. Qwen Tuned (No RAG)
- Stream/render responses to their respective cards instantly on each individual endpoint resolve.

### 4. Stylesheet adjustments: [static/style.css](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/static/style.css)
Adjust the card height and font sizes if needed to fit the new grid density comfortably. We will limit the card height to `420px` to prevent vertical page stretching.

---

## Verification Plan

### Manual Verification
1. Launch the server in the GPU environment (on the EC2 instance).
2. Submit a query (e.g. the 25% variance rule) and confirm that:
   - Skeletons for all 6 cards appear instantly.
   - The first card (Mistral Base + RAG) starts generating while the rest show `waiting...`.
   - Each card renders its response and displays its specific inference time sequentially.
   - The Grounding Reference Dashboard auto-expands on retrieval completion.
   - Comparing RAG vs. No RAG columns clearly illustrates the training alignment (toned columns follow DWP styles) and search accuracy (RAG columns contain verified citations).
