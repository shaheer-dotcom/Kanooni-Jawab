(() => {
  "use strict";

  const chatColumn = document.getElementById("chat-column");
  const welcome = document.getElementById("welcome");
  const form = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const jurisdictionSelect = document.getElementById("jurisdiction-select");
  const sendBtn = document.getElementById("send-btn");
  const newChatBtn = document.getElementById("new-chat-btn");
  const bootOverlay = document.getElementById("boot-overlay");
  const bootStatus = document.getElementById("boot-status");
  const headerMeta = document.getElementById("header-meta");
  const themeToggle = document.getElementById("theme-toggle");
  const pdfViewerOverlay = document.getElementById("pdf-viewer-overlay");
  const pdfViewerClose = document.getElementById("pdf-viewer-close");
  const pdfViewerTitle = document.getElementById("pdf-viewer-title");
  const pdfViewerStatus = document.getElementById("pdf-viewer-status");
  const pdfViewerBody = document.getElementById("pdf-viewer-body");
  const pdfPages = document.getElementById("pdf-pages");
  let pdfRenderToken = 0;

  let isStreaming = false;
  let typewriterQueue = [];
  let typewriterRunning = false;
  let activeAnswerEl = null;
  let activeCursorEl = null;

  // ── Boot: poll /health until retriever ready ──────────────────────────
  async function waitForReady() {
    const maxAttempts = 120;
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.ready) {
          enableUI();
          bootOverlay.classList.add("hidden");
          return;
        }
        if (data.error) {
          bootStatus.textContent = `Error: ${data.error}`;
          return;
        }
        bootStatus.textContent = `Loading models and connecting to Qdrant... (${i + 1}s)`;
      } catch {
        bootStatus.textContent = "Connecting to server...";
      }
      await sleep(1000);
    }
    bootStatus.textContent = "Initialization timed out. Refresh to retry.";
  }

  function enableUI() {
    queryInput.disabled = false;
    jurisdictionSelect.disabled = false;
    sendBtn.disabled = false;
    headerMeta.textContent = jurisdictionSelect.value;
    jurisdictionSelect.addEventListener("change", () => {
      headerMeta.textContent = jurisdictionSelect.value;
    });
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // ── Theme toggle ────────────────────────────────────────────────────
  function getTheme() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("kanooni-jawab-theme", theme);
  }

  function initTheme() {
    setTheme(getTheme());
    themeToggle.addEventListener("click", () => {
      setTheme(getTheme() === "dark" ? "light" : "dark");
    });
  }

  initTheme();

  // ── Message builders (Gemini alignment) ─────────────────────────────
  function hideWelcome() {
    if (welcome) welcome.style.display = "none";
  }

  function appendUserMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message user";
    el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatColumn.appendChild(el);
    scrollToBottom();
    return el;
  }

  function appendThinkingMessage() {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message thinking";
    el.innerHTML = `
      <div class="orb-loader"></div>
      <span class="thinking-step">Analyzing your query...</span>
    `;
    chatColumn.appendChild(el);
    scrollToBottom();
    return el;
  }

  function replaceThinkingWithAssistant(thinkingEl) {
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = `
      <div class="answer-text"></div>
      <div class="refs"></div>
      <div class="answer-meta"></div>
      <div class="actions">
        <button class="action-btn copy-btn" title="Copy" aria-label="Copy">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
        </button>
      </div>
    `;
    thinkingEl.replaceWith(el);
    activeAnswerEl = el.querySelector(".answer-text");
    activeCursorEl = document.createElement("span");
    activeCursorEl.className = "cursor";
    activeAnswerEl.appendChild(activeCursorEl);
    scrollToBottom();
    return el;
  }

  function appendErrorMessage(msg) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message assistant error";
    el.innerHTML = `<div class="answer-text">${escapeHtml(msg)}</div>`;
    chatColumn.appendChild(el);
    scrollToBottom();
  }

  // ── Typewriter ──────────────────────────────────────────────────────
  function enqueueTypewriter(text) {
    typewriterQueue.push(...text.split(""));
    if (!typewriterRunning) runTypewriter();
  }

  async function runTypewriter() {
    typewriterRunning = true;
    while (typewriterQueue.length > 0) {
      const backlog = typewriterQueue.length;
      const batchSize =
        backlog > 100 ? 8 :
        backlog > 40  ? 4 :
        backlog > 12  ? 2 : 1;

      let chunk = "";
      for (let i = 0; i < batchSize && typewriterQueue.length > 0; i++) {
        chunk += typewriterQueue.shift();
      }

      if (activeAnswerEl && activeCursorEl) {
        activeAnswerEl.insertBefore(document.createTextNode(chunk), activeCursorEl);
      }
      scrollToBottom();

      const delay =
        backlog > 100 ? 6 :
        backlog > 40  ? 12 :
        backlog > 12  ? 18 : 28;
      await sleep(delay);
    }
    typewriterRunning = false;
  }

  function finishTypewriter() {
    return new Promise((resolve) => {
      const check = () => {
        if (typewriterQueue.length === 0 && !typewriterRunning) {
          if (activeCursorEl) activeCursorEl.remove();
          activeCursorEl = null;
          resolve();
        } else {
          requestAnimationFrame(check);
        }
      };
      check();
    });
  }

  // ── SSE chat ────────────────────────────────────────────────────────
  async function sendQuery(query) {
    if (!query.trim() || isStreaming) return;

    isStreaming = true;
    queryInput.disabled = true;
    sendBtn.disabled = true;

    appendUserMessage(query);
    const thinkingEl = appendThinkingMessage();

    let assistantEl = null;
    let thinkingReplaced = false;
    let fullAnswer = "";

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          jurisdiction: jurisdictionSelect.value,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          if (!part.trim()) continue;

          const eventMatch = part.match(/^event:\s*(\w+)/m);
          const dataMatch = part.match(/^data:\s*(.+)$/ms);
          if (!eventMatch || !dataMatch) continue;

          const eventType = eventMatch[1];
          let data;
          try {
            data = JSON.parse(dataMatch[1]);
          } catch {
            continue;
          }

          if (eventType === "thinking") {
            const stepEl = thinkingEl.querySelector(".thinking-step");
            if (stepEl) stepEl.textContent = data.step || "Thinking...";
            scrollToBottom();
          } else if (eventType === "token") {
            if (!thinkingReplaced) {
              assistantEl = replaceThinkingWithAssistant(thinkingEl);
              thinkingReplaced = true;
            }
            const text = data.text || "";
            fullAnswer += text;
            enqueueTypewriter(text);
          } else if (eventType === "done") {
            if (!thinkingReplaced) {
              assistantEl = replaceThinkingWithAssistant(thinkingEl);
              thinkingReplaced = true;
            }
            await finishTypewriter();
            renderDone(assistantEl, data, fullAnswer || data.answer || "");
          } else if (eventType === "error") {
            thinkingEl.remove();
            appendErrorMessage(data.message || "An error occurred.");
          }
        }
      }
    } catch (err) {
      thinkingEl.remove();
      appendErrorMessage(err.message || "Failed to connect to Kanooni Jawab.");
    } finally {
      isStreaming = false;
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  }

  function renderDone(assistantEl, data, answerText) {
    if (!assistantEl) return;

    const answerEl = assistantEl.querySelector(".answer-text");
    if (answerEl && !answerEl.textContent.trim() && answerText) {
      answerEl.textContent = answerText;
    }

    const refs = data.references || [];
    const refsEl = assistantEl.querySelector(".refs");
    if (refsEl && refs.length > 0) {
      refsEl.innerHTML = refs
        .map((r, idx) => {
          const label = `SOURCE ${r.source_id} · ${r.doc_type || ""} · ${r.file_name || r.doc_name || "unknown"}`;
          if (r.has_pdf_view && r.file_url) {
            return `<button type="button" class="ref-chip ref-chip--clickable" data-ref-index="${idx}" title="Open cited passage in PDF">${escapeHtml(label)}</button>`;
          }
          return `<span class="ref-chip">${escapeHtml(label)}</span>`;
        })
        .join("");

      refsEl.querySelectorAll("[data-ref-index]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const ref = refs[parseInt(btn.dataset.refIndex, 10)];
          openPdfViewer(ref);
        });
      });
    }

    const metaEl = assistantEl.querySelector(".answer-meta");
    if (metaEl) {
      const parts = [];
      if (data.intent_type) parts.push(`Intent: ${data.intent_type}`);
      if (data.retrieval_route) parts.push(`Route: ${data.retrieval_route}`);
      if (data.retrieval_latency_ms) parts.push(`${data.retrieval_latency_ms} ms`);
      if (data.top_chunks) parts.push(`${data.top_chunks} sources`);
      metaEl.textContent = parts.join(" · ");
    }

    const copyBtn = assistantEl.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(answerText || data.answer || "").catch(() => {});
      });
    }

    scrollToBottom();
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      const shell = document.getElementById("chat-shell");
      if (shell) {
        shell.scrollTo({ top: shell.scrollHeight, behavior: "smooth" });
      }
    });
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ── PDF citation viewer ─────────────────────────────────────────────
  function closePdfViewer() {
    pdfRenderToken += 1;
    if (!pdfViewerOverlay) return;
    pdfViewerOverlay.classList.add("hidden");
    pdfViewerOverlay.setAttribute("aria-hidden", "true");
    if (pdfPages) pdfPages.innerHTML = "";
    if (pdfViewerBody) pdfViewerBody.scrollTop = 0;
    if (pdfViewerStatus) pdfViewerStatus.textContent = "";
  }

  function groupHighlightsByPage(highlights) {
    const byPage = new Map();
    for (const h of highlights) {
      const page = Number(h.page);
      if (!page || page < 1) continue;
      if (!byPage.has(page)) byPage.set(page, []);
      byPage.get(page).push(h);
    }
    return byPage;
  }

  function addHighlightBoxes(layer, pageHighlights, viewport, markScrollTarget) {
    let firstTarget = null;
    for (const h of pageHighlights) {
      const bbox = h.bbox;
      if (!bbox || bbox.length < 4) continue;
      const rect = viewport.convertToViewportRectangle(bbox);
      const left = Math.min(rect[0], rect[2]);
      const top = Math.min(rect[1], rect[3]);
      const width = Math.abs(rect[2] - rect[0]);
      const height = Math.abs(rect[3] - rect[1]);

      const div = document.createElement("div");
      div.className = "pdf-highlight";
      div.style.left = `${left}px`;
      div.style.top = `${top}px`;
      div.style.width = `${width}px`;
      div.style.height = `${height}px`;
      layer.appendChild(div);

      if (markScrollTarget && !firstTarget) {
        div.classList.add("pdf-highlight--target");
        firstTarget = div;
      }
    }
    return firstTarget;
  }

  async function openPdfViewer(ref) {
    if (!ref || !ref.file_url || !window.pdfjsLib || !pdfPages) return;

    const token = ++pdfRenderToken;
    const highlights = Array.isArray(ref.highlights) ? ref.highlights : [];
    const highlightsByPage = groupHighlightsByPage(highlights);
    const targetPage = Number(
      ref.page || (highlights[0] && highlights[0].page)
    ) || 1;

    pdfViewerOverlay.classList.remove("hidden");
    pdfViewerOverlay.setAttribute("aria-hidden", "false");
    pdfViewerTitle.textContent = ref.doc_name || ref.file_name || "Source document";
    pdfViewerStatus.textContent = "Loading PDF...";
    pdfPages.innerHTML = "";
    if (pdfViewerBody) pdfViewerBody.scrollTop = 0;

    try {
      const pdf = await pdfjsLib.getDocument(ref.file_url).promise;
      if (token !== pdfRenderToken) return;

      const containerWidth = Math.max(
        (pdfViewerBody && pdfViewerBody.clientWidth) || 720,
        320
      ) - 32;
      const samplePage = await pdf.getPage(1);
      const baseViewport = samplePage.getViewport({ scale: 1 });
      const scale = Math.min(1.5, containerWidth / baseViewport.width);

      let scrollTarget = null;
      let scrollMarked = false;
      let renderedHighlights = 0;
      const highlightPages = [...highlightsByPage.keys()].sort((a, b) => a - b);
      const firstHighlightPage = highlightPages[0] || targetPage;

      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum += 1) {
        if (token !== pdfRenderToken) return;

        pdfViewerStatus.textContent = `Rendering page ${pageNum} of ${pdf.numPages}...`;

        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale });

        const pageWrap = document.createElement("div");
        pageWrap.className = "pdf-page";
        pageWrap.dataset.page = String(pageNum);

        const canvas = document.createElement("canvas");
        const layer = document.createElement("div");
        layer.className = "pdf-highlight-layer";

        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        layer.style.width = `${viewport.width}px`;
        layer.style.height = `${viewport.height}px`;

        pageWrap.appendChild(canvas);
        pageWrap.appendChild(layer);
        pdfPages.appendChild(pageWrap);

        await page.render({
          canvasContext: canvas.getContext("2d"),
          viewport,
        }).promise;

        const pageHighlights = highlightsByPage.get(pageNum) || [];
        renderedHighlights += pageHighlights.length;
        const target = addHighlightBoxes(
          layer,
          pageHighlights,
          viewport,
          pageNum === firstHighlightPage && !scrollMarked
        );
        if (target) {
          scrollTarget = target;
          scrollMarked = true;
        }
      }

      if (token !== pdfRenderToken) return;

      pdfViewerStatus.textContent = `${pdf.numPages} page(s)${
        renderedHighlights ? ` · ${renderedHighlights} highlight(s)` : ""
      }`;

      if (scrollTarget && pdfViewerBody) {
        requestAnimationFrame(() => {
          const bodyRect = pdfViewerBody.getBoundingClientRect();
          const targetRect = scrollTarget.getBoundingClientRect();
          const offset =
            targetRect.top -
            bodyRect.top +
            pdfViewerBody.scrollTop -
            bodyRect.height * 0.35;
          pdfViewerBody.scrollTo({
            top: Math.max(0, offset),
            behavior: "smooth",
          });
        });
      } else if (targetPage > 1 && pdfViewerBody) {
        const pageEl = pdfPages.querySelector(`[data-page="${targetPage}"]`);
        if (pageEl) {
          requestAnimationFrame(() => {
            pageEl.scrollIntoView({ behavior: "smooth", block: "start" });
          });
        }
      }
    } catch (err) {
      if (token !== pdfRenderToken) return;
      pdfViewerStatus.textContent = `Failed to load PDF: ${err.message || err}`;
    }
  }

  if (pdfViewerClose) {
    pdfViewerClose.addEventListener("click", closePdfViewer);
  }
  if (pdfViewerOverlay) {
    pdfViewerOverlay.addEventListener("click", (e) => {
      if (e.target === pdfViewerOverlay) closePdfViewer();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePdfViewer();
  });

  // ── Events ──────────────────────────────────────────────────────────
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (!q) return;
    queryInput.value = "";
    sendQuery(q);
  });

  newChatBtn.addEventListener("click", () => {
    if (isStreaming) return;
    chatColumn.innerHTML = "";
    if (welcome) {
      welcome.style.display = "";
      chatColumn.appendChild(welcome);
    }
    queryInput.focus();
  });

  document.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q;
      if (q) sendQuery(q);
    });
  });

  waitForReady();
})();
