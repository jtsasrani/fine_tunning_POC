/**
 * DWP CMS Decision Support Suite v2.0
 * Frontend Controller Script
 */

document.addEventListener("DOMContentLoaded", () => {
    // API Authorization Key
    const getApiKey = () => document.getElementById("api-key-input").value.strip || document.getElementById("api-key-input").value;
    const authHeaders = () => ({
        "Content-Type": "application/json",
        "X-API-Key": getApiKey()
    });

    // Conversation History Storage
    let conversationMessages = [];
    
    // Chart Instance reference
    let convergenceChart = null;

    // -------------------------------------------------------------
    // 1. Navigation Panel Controls
    // -------------------------------------------------------------
    const navItems = document.querySelectorAll(".nav-item");
    const panelSections = document.querySelectorAll(".panel-section");
    const panelTitle = document.getElementById("panel-title");
    const panelDescription = document.getElementById("panel-description");

    const panelMeta = {
        "chat-panel": {
            title: "AI Policy Assistant",
            desc: "Ask complex policy and procedural questions directly aligned with CMS guidelines."
        },
        "dashboard-panel": {
            title: "Training Performance Dashboard",
            desc: "Real-time metrics tracking the training of the Qwen-14B student model."
        },
        "rag-panel": {
            title: "RAG Passage Explorer",
            desc: "Search, inspect, and analyze document chunks retrieved from the FAISS database."
        }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const target = item.getAttribute("data-target");
            
            // Switch navigation active state
            navItems.forEach(n => n.classList.remove("active"));
            item.classList.add("active");
            
            // Switch panel active state
            panelSections.forEach(p => p.classList.remove("active"));
            document.getElementById(target).classList.add("active");
            
            // Update Headers
            panelTitle.textContent = panelMeta[target].title;
            panelDescription.textContent = panelMeta[target].desc;
            
            // Trigger specific actions based on panel load
            if (target === "dashboard-panel") {
                fetchAndRenderDashboardMetrics();
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Formatting Helpers
    // -------------------------------------------------------------
    const parseMarkdown = (text) => {
        if (!text) return "";
        let html = text;
        
        // Escape HTML tags to prevent XSS but keep custom markdown
        html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        
        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        
        // Blockquotes: > text
        html = html.replace(/^>\s+(.*)$/gim, "<blockquote>$1</blockquote>");
        
        // Bullet list points: - text or * text
        html = html.replace(/^\s*[-*]\s+(.*)$/gim, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
        
        // Code quotes: `code`
        html = html.replace(/`(.*?)`/g, "<code>$1</code>");
        
        // Newlines to line breaks
        html = html.replace(/\n/g, "<br>");
        
        return html;
    };

    // -------------------------------------------------------------
    // 3. AI Chatbot Logic
    // -------------------------------------------------------------
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatBox = document.getElementById("chat-box");
    const welcomeScreen = document.getElementById("welcome-screen");
    const citationSidebar = document.getElementById("citation-sidebar");
    const closeCitations = document.getElementById("close-citations");
    const citationsContainer = document.getElementById("citations-container");
    const modelSelect = document.getElementById("model-select");
    const ragToggle = document.getElementById("rag-toggle");

    // Close Citations Sidebar
    closeCitations.addEventListener("click", () => {
        citationSidebar.classList.add("closed");
    });

    // Reset Chat Button Handler
    const resetChatBtn = document.getElementById("new-chat-btn");
    if (resetChatBtn) {
        resetChatBtn.addEventListener("click", () => {
            conversationMessages = [];
            // Remove message bubbles
            const bubbles = chatBox.querySelectorAll(".message-bubble");
            bubbles.forEach(b => b.remove());
            // Restore welcome screen
            if (welcomeScreen) welcomeScreen.style.display = "block";
            // Clear citations
            citationsContainer.innerHTML = `
                <div class="no-citations">
                    <i class="fa-solid fa-magnifying-glass-chart"></i>
                    <p>Context citations will appear here once you send a query in RAG mode.</p>
                </div>
            `;
            // Close citations sidebar
            if (citationSidebar) citationSidebar.classList.add("closed");
            
            chatInput.value = "";
            chatInput.style.height = "auto";
        });
    }

    // Textarea Auto-expand
    chatInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // Suggestion Cards click handler
    document.querySelectorAll(".suggestion-card").forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            chatInput.value = prompt;
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // Render Message bubble
    const appendMessage = (role, content, responseTime = null) => {
        // Remove welcome screen if present
        if (welcomeScreen) welcomeScreen.style.display = "none";
        
        const bubble = document.createElement("div");
        bubble.classList.add("message-bubble", role);
        
        const avatar = document.createElement("div");
        avatar.classList.add("message-avatar");
        avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const container = document.createElement("div");
        container.classList.add("message-content-wrapper");
        
        const messageBody = document.createElement("div");
        messageBody.classList.add("message-content");
        messageBody.innerHTML = role === "user" ? parseMarkdown(content) : parseMarkdown(content);
        
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const timeElem = document.createElement("span");
        timeElem.classList.add("message-time");
        timeElem.textContent = responseTime ? `${timeStr} • took ${responseTime}s` : timeStr;
        
        container.appendChild(messageBody);
        container.appendChild(timeElem);
        bubble.appendChild(avatar);
        bubble.appendChild(container);
        
        chatBox.appendChild(bubble);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        return bubble;
    };

    // Render Typing Indicator
    const showTypingIndicator = () => {
        const indicator = document.createElement("div");
        indicator.classList.add("message-bubble", "assistant", "indicator-bubble");
        
        const avatar = document.createElement("div");
        avatar.classList.add("message-avatar");
        avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
        
        const content = document.createElement("div");
        content.classList.add("message-content");
        content.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        
        indicator.appendChild(avatar);
        indicator.appendChild(content);
        chatBox.appendChild(indicator);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        return indicator;
    };

    // Chat submit handler
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        
        // Reset input box height
        chatInput.value = "";
        chatInput.style.height = "auto";
        
        // Append user query to logs and screen
        appendMessage("user", text);
        conversationMessages.push({ role: "user", content: text });
        
        const typingBubble = showTypingIndicator();
        
        try {
            let contexts = [];
            
            if (ragToggle.checked) {
                const retrieveResponse = await fetch("/api/query/retrieve", {
                    method: "POST",
                    headers: authHeaders(),
                    body: JSON.stringify({ 
                        query: text,
                        history: conversationMessages,
                        model: modelSelect.value
                    })
                });
                
                if (retrieveResponse.ok) {
                    const retrieveResult = await retrieveResponse.json();
                    contexts = retrieveResult.retrieved_contexts || [];
                    renderCitations(contexts);
                } else {
                    console.error("Context retrieval failed.");
                }
            } else {
                // Clear citations pane if RAG is disabled
                citationsContainer.innerHTML = `
                    <div class="no-citations">
                        <i class="fa-solid fa-ban"></i>
                        <p>Hybrid RAG is disabled. Answers are based entirely on pre-trained parameters.</p>
                    </div>
                `;
            }
            
            // 2. Answer Generation Step
            const generateResponse = await fetch("/api/query/generate", {
                method: "POST",
                headers: authHeaders(),
                body: JSON.stringify({
                    messages: conversationMessages,
                    query: text,
                    model: modelSelect.value,
                    contexts: contexts,
                    use_rag: ragToggle.checked
                })
            });
            
            // Remove typing bubble
            typingBubble.remove();
            
            if (generateResponse.ok) {
                const generateResult = await generateResponse.json();
                const aiAnswer = generateResult.response || "No response received.";
                
                appendMessage("assistant", aiAnswer, generateResult.time);
                conversationMessages.push({ role: "assistant", content: aiAnswer });
            } else {
                const err = await generateResponse.json();
                appendMessage("assistant", `**Error**: Failed to generate response (${err.error || "Unknown server error"}).`);
            }
            
        } catch (error) {
            console.error("Error sending query:", error);
            typingBubble.remove();
            appendMessage("assistant", `**Connection Error**: Failed to connect to backend server.`);
        }
    });

    // Render citations inside sidebar
    const renderCitations = (contexts) => {
        if (!contexts || contexts.length === 0) {
            citationsContainer.innerHTML = `
                <div class="no-citations">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>No highly relevant passages could be retrieved for this query.</p>
                </div>
            `;
            return;
        }
        
        // Add a small info box at the top explaining the filtering logic
        citationsContainer.innerHTML = `
            <div class="citations-info">
                <i class="fa-solid fa-circle-info"></i>
                <p>Passages with a Rerank score &lt; -1.0 are kept here for inspection but are excluded from the AI reader context to prevent hallucination.</p>
            </div>
        `;
        
        contexts.forEach((c) => {
            const card = document.createElement("div");
            card.classList.add("citation-card");
            
            const isExcluded = c.score < -1.0;
            if (isExcluded) {
                card.classList.add("excluded");
            }
            
            const scoreText = isExcluded ? `Rerank: ${c.score} (Excluded)` : `Rerank: ${c.score}`;
            const badgeClass = isExcluded ? "score-badge excluded" : "score-badge";
            
            card.innerHTML = `
                <div class="citation-source">
                    <a href="/api/documents/${encodeURIComponent(c.source_doc)}?api_key=${encodeURIComponent(getApiKey())}" target="_blank" class="source-badge clickable-source" title="Open ${c.source_doc} in a new tab">${c.source_doc}</a>
                    <span class="${badgeClass}">${scoreText}</span>
                </div>
                <div class="citation-text">
                    <strong>Ref #${c.paragraph_id.split('_').pop()}</strong>: ${c.text.substring(0, 200)}...
                </div>
            `;
            citationsContainer.appendChild(card);
        });
        
        // Open the citations sidebar
        citationSidebar.classList.remove("closed");
    };

    // -------------------------------------------------------------
    // 4. Training Dashboard Metrics
    // -------------------------------------------------------------
    const currentLossVal = document.getElementById("current-loss-val");
    const valLossVal = document.getElementById("val-loss-val");
    const trainingBadge = document.getElementById("training-badge");
    const validationTableBody = document.querySelector("#validation-history-table tbody");
    
    // Status text in sidebar
    const modeText = document.getElementById("mode-text");
    const progressText = document.getElementById("training-progress-text");
    const pipelineText = document.getElementById("pipeline-status-text");

    const fetchAndRenderDashboardMetrics = async () => {
        try {
            const response = await fetch("/api/metrics");
            if (!response.ok) throw new Error("Metrics endpoint failed");
            
            const metrics = await response.json();
            const summary = metrics.summary || {};
            const trainHistory = metrics.train_history || [];
            const evalHistory = metrics.eval_history || [];
            
            // 1. Update KPI Values
            currentLossVal.textContent = summary.final_loss ? summary.final_loss.toFixed(4) : "--";
            valLossVal.textContent = summary.final_eval_loss ? summary.final_eval_loss.toFixed(4) : "--";
            
            // Status Indicator update
            if (summary.status === "running" && summary.completed_steps < 2400) {
                trainingBadge.textContent = "Active Running";
                trainingBadge.className = "badge pulse";
                document.getElementById("sys-indicator").className = "status-indicator online";
                pipelineText.textContent = "Training Fine-Tuning";
            } else if (summary.status === "completed" || summary.completed_steps >= 2400) {
                trainingBadge.textContent = "Completed";
                trainingBadge.className = "badge";
                document.getElementById("sys-indicator").className = "status-indicator online";
                pipelineText.textContent = "Run Completed";
            } else {
                trainingBadge.textContent = "Idle";
                trainingBadge.className = "badge";
                pipelineText.textContent = "Idle / Idle";
            }
            if (progressText) {
                progressText.textContent = `${summary.completed_steps || 0}/${summary.total_steps || 2406}`;
            }
            
            // 2. Render Checkpoint Table
            if (evalHistory.length === 0) {
                validationTableBody.innerHTML = `
                    <tr>
                        <td colspan="4" class="empty-table">No validation cycles processed yet. Checkpoints run every 100 steps.</td>
                    </tr>
                `;
            } else {
                validationTableBody.innerHTML = "";
                // Render list in reverse order (newest first)
                [...evalHistory].reverse().forEach(ev => {
                    const row = document.createElement("tr");
                    // Find corresponding training loss at this step
                    const trainLossItem = trainHistory.find(t => t.step === ev.step) || trainHistory.find(t => Math.abs(t.step - ev.step) <= 10);
                    const trainLossText = trainLossItem ? trainLossItem.loss.toFixed(4) : "N/A";
                    row.innerHTML = `
                        <td><strong>Step ${ev.step}</strong></td>
                        <td>Epoch ${ev.epoch.toFixed(4)}</td>
                        <td><span class="highlight-purple">${trainLossText}</span></td>
                        <td><span class="highlight-green">${ev.eval_loss.toFixed(4)}</span></td>
                    `;
                    validationTableBody.appendChild(row);
                });
            }
            
            // 3. Render Chart
            renderConvergenceChart(trainHistory, evalHistory);
            
        } catch (error) {
            console.error("Error loading metrics dashboard:", error);
        }
    };

    const renderConvergenceChart = (trainHist, evalHist) => {
        const ctx = document.getElementById("convergenceChart").getContext("2d");
        
        // Prepare training loss coordinates (sample every 10 steps)
        const trainData = trainHist.map(h => ({ x: h.step, y: h.loss }));
        
        // Prepare validation loss coordinates (sample every 100 steps)
        const evalData = evalHist.map(h => ({ x: h.step, y: h.eval_loss }));

        if (convergenceChart) {
            // Update existing chart to prevent canvas duplicate overlaps
            convergenceChart.data.datasets[0].data = trainData;
            convergenceChart.data.datasets[1].data = evalData;
            convergenceChart.update();
            return;
        }

        convergenceChart = new Chart(ctx, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Training Loss",
                        data: trainData,
                        borderColor: "#8b5cf6", // Neon Violet
                        backgroundColor: "rgba(139, 92, 246, 0.05)",
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: "Validation Loss",
                        data: evalData,
                        borderColor: "#06b6d4", // Neon Cyan
                        backgroundColor: "rgba(6, 182, 212, 0.2)",
                        borderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        showLine: true,
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Training Steps',
                            color: '#9ca3af'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.03)'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss (Cross Entropy)',
                            color: '#9ca3af'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.03)'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#f3f4f6'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    };

    // -------------------------------------------------------------
    // 5. RAG Passage Explorer
    // -------------------------------------------------------------
    const explorerForm = document.getElementById("rag-explorer-form");
    const explorerInput = document.getElementById("rag-query-input");
    const passagesContainer = document.getElementById("explorer-passages-container");
    const explorerTimeVal = document.getElementById("explorer-time-val");
    const explorerScoreVal = document.getElementById("explorer-score-val");

    explorerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = explorerInput.value.trim();
        if (!query) return;
        
        passagesContainer.innerHTML = `
            <div class="no-passages">
                <div class="typing-indicator"><span></span><span></span><span></span></div>
                <p>Retrieving policy passages...</p>
            </div>
        `;
        
        try {
            const response = await fetch("/api/query/retrieve", {
                method: "POST",
                headers: authHeaders(),
                body: JSON.stringify({ query: query })
            });
            
            if (response.ok) {
                const data = await response.json();
                const contexts = data.retrieved_contexts || [];
                
                explorerTimeVal.textContent = `${data.retrieval_time}s`;
                
                if (contexts.length === 0) {
                    explorerScoreVal.textContent = "--";
                    passagesContainer.innerHTML = `
                        <div class="no-passages">
                            <i class="fa-solid fa-box-open"></i>
                            <p>No matching passages found. Try a different terminology.</p>
                        </div>
                    `;
                    return;
                }
                
                explorerScoreVal.textContent = contexts[0].score.toFixed(4);
                
                // Add informational banner about RAG filtering
                passagesContainer.innerHTML = `
                    <div class="citations-info">
                        <i class="fa-solid fa-circle-info"></i>
                        <p>Passages with a Rerank score &lt; -1.0 are shown here for exploration, but are automatically excluded from the AI reader context to prevent hallucinations.</p>
                    </div>
                `;
                
                contexts.forEach((c) => {
                    const card = document.createElement("div");
                    card.classList.add("explorer-passage-card");
                    
                    const isExcluded = c.score < -1.0;
                    if (isExcluded) {
                        card.classList.add("excluded");
                    }
                    
                    const rerankText = isExcluded ? `Rerank: ${c.score} (Excluded)` : `Rerank: ${c.score}`;
                    
                    card.innerHTML = `
                        <div class="explorer-passage-meta">
                            <a href="/api/documents/${encodeURIComponent(c.source_doc)}?api_key=${encodeURIComponent(getApiKey())}" target="_blank" class="explorer-source-badge clickable-source" title="Open ${c.source_doc} in a new tab">${c.source_doc}</a>
                            <span class="explorer-score-tag ${isExcluded ? 'excluded' : ''}">${rerankText} (Ref #${c.paragraph_id.split('_').pop()})</span>
                        </div>
                        <div class="explorer-passage-text">
                            ${c.text}
                        </div>
                    `;
                    passagesContainer.appendChild(card);
                });
            } else {
                const err = await response.json();
                passagesContainer.innerHTML = `
                    <div class="no-passages">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <p>Error running RAG explorer: ${err.error || "Unknown server error"}</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error("Error retrieving explorer passages:", error);
            passagesContainer.innerHTML = `
                <div class="no-passages">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>Connection Error: Failed to retrieve data from server.</p>
                </div>
            `;
        }
    });

    // -------------------------------------------------------------
    // 6. System Status Auto-Check
    // -------------------------------------------------------------
    const updateSystemStatusInfo = async () => {
        try {
            const response = await fetch("/api/metrics");
            if (response.ok) {
                const data = await response.json();
                const summary = data.summary || {};
                
                // Determine model availability and mode based on actual backend operational state
                const activeModel = modelSelect ? modelSelect.value : "qwen_14b_tuned";
                let hasActiveModel = false;
                let displayName = "";
                
                if (activeModel === "qwen_14b_tuned") {
                    hasActiveModel = !data.demo_mode && data.models_loaded && data.models_loaded.includes("qwen_14b_tuned");
                    displayName = "Qwen-14B";
                } else if (activeModel === "dwp-cmg-llama-8b-endpoint-v2") {
                    hasActiveModel = !data.demo_mode && data.models_loaded && data.models_loaded.includes("dwp-cmg-llama-8b-endpoint-v2");
                    displayName = "Llama-8B";
                }
                
                modeText.textContent = hasActiveModel ? `Inference Active (${displayName})` : `Demo Mode (${displayName} Simulated)`;
                modeText.style.color = hasActiveModel ? "#34d399" : "#c084fc";
                if (progressText) {
                    progressText.textContent = `${summary.completed_steps || 0}/${summary.total_steps || 2406}`;
                }
                
                if (summary.status === "completed" || summary.completed_steps >= 2400) {
                    pipelineText.textContent = "Pipeline Complete";
                    pipelineText.style.color = "#34d399";
                } else if (summary.status === "running") {
                    pipelineText.textContent = "Fine-Tuning Active";
                    pipelineText.style.color = "#a78bfa";
                } else if (summary.status === "failed") {
                    pipelineText.textContent = "Pipeline Failed";
                    pipelineText.style.color = "#f43f5e";
                } else {
                    pipelineText.textContent = "Idle";
                    pipelineText.style.color = "#9ca3af";
                }
                
                document.getElementById("sys-indicator").className = "status-indicator online";
            }
        } catch (error) {
            document.getElementById("sys-indicator").className = "status-indicator offline";
            modeText.textContent = "Disconnected";
            modeText.style.color = "#f43f5e";
        }
    };

    // Listen for model select dropdown changes to update system status in real-time
    if (modelSelect) {
        modelSelect.addEventListener("change", () => {
            updateSystemStatusInfo();
        });
    }

    // Run status check immediately and then every 20 seconds
    updateSystemStatusInfo();
    setInterval(updateSystemStatusInfo, 20000);
});
