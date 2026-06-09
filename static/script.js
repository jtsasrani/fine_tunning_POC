document.addEventListener("DOMContentLoaded", () => {
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");
    
    // Quick starts
    const quickStartBtns = document.querySelectorAll(".quick-start-btn");
    
    // Skeletons
    const skeletons = {
        a: document.getElementById("skeleton-a"),
        b: document.getElementById("skeleton-b"),
        c: document.getElementById("skeleton-c"),
        d: document.getElementById("skeleton-d"),
        e: document.getElementById("skeleton-e"),
        f: document.getElementById("skeleton-f")
    };
    
    // Contents
    const contents = {
        a: document.getElementById("content-a"),
        b: document.getElementById("content-b"),
        c: document.getElementById("content-c"),
        d: document.getElementById("content-d"),
        e: document.getElementById("content-e"),
        f: document.getElementById("content-f")
    };
    
    // Times
    const times = {
        a: document.getElementById("time-a"),
        b: document.getElementById("time-b"),
        c: document.getElementById("time-c"),
        d: document.getElementById("time-d"),
        e: document.getElementById("time-e"),
        f: document.getElementById("time-f")
    };
    
    // Copy buttons
    const copyBtns = document.querySelectorAll(".copy-btn");
    
    // Accordion
    const retrievalSection = document.getElementById("retrieval-section");
    const toggleRetrievalBtn = document.getElementById("toggle-retrieval-btn");
    const retrievalContent = document.getElementById("retrieval-content");
    const retrievalTimeBadge = document.getElementById("retrieval-time");
    const retrievedChunksContainer = document.getElementById("retrieved-chunks-container");

    // Click event for quick start pills
    quickStartBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            queryInput.value = btn.getAttribute("data-query");
            queryForm.dispatchEvent(new Event("submit"));
        });
    });

    // Toggle retrieval accordion
    toggleRetrievalBtn.addEventListener("click", () => {
        retrievalSection.classList.toggle("collapsed");
        retrievalContent.classList.toggle("hidden");
    });

    // Copy to clipboard
    copyBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            const textElement = document.getElementById(targetId);
            if (!textElement) return;
            
            // Get original plain text (ignoring HTML spans)
            const textToCopy = textElement.innerText;
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                btn.classList.add("copied");
                const originalHtml = btn.innerHTML;
                
                // Show success icon (checkmark)
                btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><polyline points="20 6 9 17 4 12"/></svg>`;
                
                setTimeout(() => {
                    btn.classList.remove("copied");
                    btn.innerHTML = originalHtml;
                }, 2000);
            }).catch(err => {
                console.error("Could not copy text: ", err);
            });
        });
    });

    // Process query submission
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Set Loading State
        submitBtn.disabled = true;
        queryInput.disabled = true;
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");

        // Hide copy buttons during load
        copyBtns.forEach(btn => btn.classList.add("hidden"));

        // Show card skeletons, clear content classes & set to waiting
        Object.keys(skeletons).forEach(key => {
            skeletons[key].classList.remove("hidden");
            contents[key].classList.add("empty-state");
            contents[key].innerHTML = "";
            times[key].textContent = "waiting...";
        });

        // Reset grounding dashboard
        toggleRetrievalBtn.disabled = true;
        retrievalSection.classList.add("collapsed");
        retrievalContent.classList.add("hidden");
        retrievalTimeBadge.textContent = "Retrieved: 0 chunks (0.00s)";
        retrievedChunksContainer.innerHTML = "";

        try {
            // Step 1: Retrieve context chunks first (takes < 0.1s)
            const retrieveResponse = await fetch("/api/query/retrieve", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query })
            });

            if (!retrieveResponse.ok) {
                throw new Error(`Retrieval failed with code ${retrieveResponse.status}`);
            }

            const retrieveData = await retrieveResponse.json();
            if (retrieveData.error) {
                throw new Error(retrieveData.error);
            }

            // Populate and show the grounding dashboard immediately
            renderRetrievedChunks(retrieveData.retrieved_contexts, retrieveData.retrieval_time);
            toggleRetrievalBtn.disabled = false;
            retrievalSection.classList.remove("collapsed");
            retrievalContent.classList.remove("hidden");

            // Step 2: Generate answers sequentially (to avoid GPU hardware contention)
            // Sequence: Mistral Base + RAG -> Mistral Base (No RAG) -> Mistral Tuned + RAG -> Mistral Tuned (No RAG) -> Qwen Tuned + RAG -> Qwen Tuned (No RAG)
            const modelsToRun = [
                { key: "c", id: "mistral_base", label: "Mistral Base + RAG", use_rag: true },
                { key: "d", id: "mistral_base", label: "Mistral Base (No RAG)", use_rag: false },
                { key: "a", id: "mistral_tuned", label: "Mistral Tuned + RAG", use_rag: true },
                { key: "e", id: "mistral_tuned", label: "Mistral Tuned (No RAG)", use_rag: false },
                { key: "b", id: "qwen_tuned", label: "Qwen Tuned + RAG", use_rag: true },
                { key: "f", id: "qwen_tuned", label: "Qwen Tuned (No RAG)", use_rag: false }
            ];

            for (const model of modelsToRun) {
                times[model.key].textContent = "generating...";
                try {
                    const genResponse = await fetch("/api/query/generate", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            query: query,
                            model: model.id,
                            use_rag: model.use_rag,
                            contexts: model.use_rag ? retrieveData.retrieved_contexts : []
                        })
                    });

                    if (!genResponse.ok) {
                        throw new Error(`${model.label} failed with code ${genResponse.status}`);
                    }

                    const genData = await genResponse.json();
                    if (genData.error) {
                        throw new Error(genData.error);
                    }

                    // Render result immediately on completion
                    renderConfigResult(model.key, genData.response, genData.time);
                } catch (genError) {
                    console.error(genError);
                    skeletons[model.key].classList.add("hidden");
                    contents[model.key].classList.add("empty-state");
                    contents[model.key].innerHTML = `<p style="color: var(--color-pink)">Error: ${genError.message}</p>`;
                    times[model.key].textContent = "error";
                }
            }

        } catch (error) {
            console.error(error);
            Object.keys(contents).forEach(key => {
                skeletons[key].classList.add("hidden");
                contents[key].classList.add("empty-state");
                contents[key].innerHTML = `<p style="color: var(--color-pink)">Error generating response: ${error.message}</p>`;
                times[key].textContent = "error";
            });
        } finally {
            // Restore Submit State
            submitBtn.disabled = false;
            queryInput.disabled = false;
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
    });

    // Helper: Style output texts dynamically
    function formatResponseText(text) {
        if (!text) return "No response generated.";
        
        let html = text;
        
        // Escape HTML tags to prevent injections, but preserve formatting additions
        html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        // Format rules/terms e.g., "Paragraph 17" or "chapter 25"
        html = html.replace(/(Paragraph\s+\d+(?:[a-zA-Z])?|Para\s+\d+|Chapter\s+\d+|Table\s+\d+)/gi, '<span class="citation">$1</span>');

        // Format figures (e.g. 25%, £100, £3,000, 31 August, 0.25)
        html = html.replace(/(\b\d+(?:\.\d+)?%\b|\b\d+\s*percent\b|£\d+(?:,\d+)*(?:\.\d+)?)/gi, '<span class="number-highlight">$1</span>');

        // Line breaks to <br>
        html = html.replace(/\n/g, "<br>");

        return html;
    }

    // Helper: Render config matrix data
    function renderConfigResult(key, response, time) {
        skeletons[key].classList.add("hidden");
        contents[key].classList.remove("empty-state");
        contents[key].innerHTML = formatResponseText(response);
        times[key].textContent = `${time}s`;
        
        // Reveal copy button
        const copyBtn = document.querySelector(`.copy-btn[data-target="content-${key}"]`);
        if (copyBtn) {
            copyBtn.classList.remove("hidden");
        }
    }

    // Helper: Render chunks in reference drawer
    function renderRetrievedChunks(chunks, time) {
        retrievalTimeBadge.textContent = `Retrieved: ${chunks.length} chunks (${time}s)`;
        
        if (!chunks || chunks.length === 0) {
            retrievedChunksContainer.innerHTML = `<p class="intro-text">No chunks retrieved.</p>`;
            return;
        }
        
        chunks.forEach((chunk, index) => {
            const chunkCard = document.createElement("div");
            chunkCard.className = "chunk-card";
            chunkCard.innerHTML = `
                <div class="chunk-meta">
                    <span class="chunk-source">Doc: ${chunk.source_doc} (Para ${chunk.paragraph_id})</span>
                    <span class="score-badge" title="Hybrid Score: 0.5 * Vector Similarity + 0.5 * BM25 Keyword Match">Score: ${chunk.score}</span>
                </div>
                <div class="chunk-text">
                    ${chunk.text}
                </div>
            `;
            retrievedChunksContainer.appendChild(chunkCard);
        });
    }
});
