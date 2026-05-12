/**
 * FilingLens AI  Frontend JavaScript
 * ======================================
 * Handles:
 *   - Drag & drop file upload
 *   - EDGAR / upload source switching
 *   - Analysis request lifecycle
 *   - Loading state animation
 *   - Results rendering
 *   - Sentiment distribution chart
 *   - Chunk breakdown list
 *   - Navbar scroll effect
 */

"use strict";

/*  State  */
let currentSource = "upload";   // "upload" | "edgar"
let uploadedFilename = null;    // set after successful file upload
let isAnalyzing = false;

/*  DOM References  */
const $ = (id) => document.getElementById(id);

/*  Init  */
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initDropZone();
  initFileInput();
  initTerminalTyping();
});

/*  Navbar scroll effect  */
function initNavbar() {
  const navbar = $("navbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  }, { passive: true });
}

/*  Source Toggle  */
function switchSource(source) {
  currentSource = source;

  const btnUpload = $("btn-upload");
  const btnEdgar  = $("btn-edgar");
  const secUpload = $("section-upload");
  const secEdgar  = $("section-edgar");

  if (source === "upload") {
    btnUpload.classList.add("active");
    btnEdgar.classList.remove("active");
    secUpload.style.display = "block";
    secEdgar.style.display  = "none";
  } else {
    btnEdgar.classList.add("active");
    btnUpload.classList.remove("active");
    secUpload.style.display = "none";
    secEdgar.style.display  = "block";
  }
}

/*  Drop Zone  */
function initDropZone() {
  const zone = $("drop-zone");
  if (!zone) return;

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    zone.addEventListener(evt, preventDefaults, false);
    document.body.addEventListener(evt, preventDefaults, false);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    zone.addEventListener(evt, () => zone.classList.add("dragover"));
  });

  ["dragleave", "drop"].forEach((evt) => {
    zone.addEventListener(evt, () => zone.classList.remove("dragover"));
  });

  zone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file) handleFileSelect(file);
  });

  zone.addEventListener("click", () => $("file-input").click());
}

function initFileInput() {
  const input = $("file-input");
  if (!input) return;
  input.addEventListener("change", () => {
    if (input.files[0]) handleFileSelect(input.files[0]);
  });
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

/**
 * Upload the selected file to the backend and store the filename.
 */
async function handleFileSelect(file) {
  const allowed = ["text/plain", "text/html"];
  const allowedExt = [".txt", ".html", ".htm"];
  const ext = "." + file.name.split(".").pop().toLowerCase();

  if (!allowedExt.includes(ext)) {
    showToast("Unsupported file type. Please use .txt, .html, or .htm.", "error");
    return;
  }

  showToast("Uploading file", "info");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      showToast(data.error || "Upload failed.", "error");
      return;
    }

    uploadedFilename = data.filename;
    $("upload-filename").textContent = file.name;
    $("upload-status").style.display = "block";
    showToast("File uploaded successfully.", "success");

  } catch (err) {
    console.error("Upload error:", err);
    showToast("Network error during upload.", "error");
  }
}

function clearUpload() {
  uploadedFilename = null;
  $("upload-status").style.display = "none";
  $("file-input").value = "";
}

/*  Analysis  */
async function runAnalysis() {
  if (isAnalyzing) return;

  const ticker      = ($("ticker-input").value || "").trim().toUpperCase();
  const filingType  = $("filing-type").value;
  const section     = ($("section-input").value || "").trim();

  // Validation
  if (currentSource === "upload" && !uploadedFilename) {
    showToast("Please upload a filing file first.", "error");
    return;
  }
  if (currentSource === "edgar" && !ticker) {
    showToast("Please enter a ticker symbol.", "error");
    return;
  }

  isAnalyzing = true;
  setAnalyzeBtn(true);
  showLoading();

  // Animate loading steps with delays
  animateLoadingSteps();

  const payload = {
    section,
    ...(currentSource === "upload"
      ? { filename: uploadedFilename }
      : { ticker, filing_type: filingType }),
  };

  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showToast(data.error || "Analysis failed.", "error");
      showEmpty();
      return;
    }

    renderResults(data);
    smoothScrollTo("results-panel");

  } catch (err) {
    console.error("Analysis error:", err);
    showToast("Network error. Is the server running?", "error");
    showEmpty();
  } finally {
    isAnalyzing = false;
    setAnalyzeBtn(false);
  }
}

/*  Loading Animation  */
const LOADING_STEPS = [
  { id: "lstep-1", delay: 0,    title: "Extracting text",          sub: "Parsing and cleaning the filing content." },
  { id: "lstep-2", delay: 2000, title: "Chunking document",        sub: "Splitting into model-compatible segments." },
  { id: "lstep-3", delay: 4000, title: "Running local inference",  sub: "Qwen2-0.5B is analyzing each chunk. This may take several minutes." },
  { id: "lstep-4", delay: 8000, title: "Aggregating results",      sub: "Computing majority-vote sentiment across all chunks." },
];

let loadingTimers = [];

function animateLoadingSteps() {
  // Reset
  loadingTimers.forEach(clearTimeout);
  loadingTimers = [];
  LOADING_STEPS.forEach(({ id }) => {
    const el = $(id);
    el.classList.remove("active", "done");
  });

  LOADING_STEPS.forEach(({ id, delay, title, sub }, idx) => {
    const timer = setTimeout(() => {
      // Mark previous as done
      if (idx > 0) {
        const prev = $(LOADING_STEPS[idx - 1].id);
        prev.classList.remove("active");
        prev.classList.add("done");
      }
      const el = $(id);
      el.classList.add("active");
      $("loading-title").textContent = title;
      $("loading-sub").textContent   = sub;
    }, delay);
    loadingTimers.push(timer);
  });
}

/*  Results Rendering  */
function renderResults(data) {
  const sentiment = data.overall_sentiment; // "Positive" | "Neutral" | "Negative"

  //  Sentiment Badge
  const badge = $("sentiment-badge");
  badge.className = "sentiment-badge " + sentiment.toLowerCase();

  const iconMap = { Positive: "", Neutral: "", Negative: "" };
  $("sentiment-icon").textContent = iconMap[sentiment] || "";
  $("sentiment-label").textContent = sentiment;

  //  Meta
  $("res-source").textContent = data.source || "";
  $("res-ticker").textContent = data.ticker || "";
  $("res-type").textContent   = data.filing_type || "";
  $("res-words").textContent  = (data.word_count || 0).toLocaleString() + " words";
  $("res-chunks").textContent = data.chunk_count + " chunks";

  //  EDGAR link
  if (data.filing_url) {
    $("edgar-link").href = data.filing_url;
    $("edgar-link-row").style.display = "block";
  } else {
    $("edgar-link-row").style.display = "none";
  }

  //  Sentiment Hero card border
  const heroCard = $("sentiment-hero-card");
  const colorMap = {
    Positive: "var(--accent-green)",
    Negative: "var(--accent-red)",
    Neutral:  "var(--accent-yellow)",
  };
  heroCard.style.borderLeftColor = colorMap[sentiment] || "var(--accent-primary)";

  //  Distribution
  renderDistribution(data.distribution || {});

  //  Explanation & Summary
  $("res-explanation").textContent = data.explanation || "";
  $("res-summary").textContent     = data.summary || "";

  //  Text Preview
  $("text-preview").textContent = data.text_preview || "";

  //  Chunk List
  renderChunkList(data.chunk_results || []);

  showResults();
}

/**
 * Build the three sentiment distribution bars.
 */
function renderDistribution(dist) {
  const container = $("distribution-chart");
  container.innerHTML = "";

  const sentiments = [
    { key: "Positive", cls: "positive" },
    { key: "Neutral",  cls: "neutral" },
    { key: "Negative", cls: "negative" },
  ];

  sentiments.forEach(({ key, cls }) => {
    const pct = dist[key] ?? 0;

    const row = document.createElement("div");
    row.className = "dist-bar-row";

    row.innerHTML = `
      <span class="dist-label ${cls}">${key}</span>
      <div class="dist-bar-track">
        <div class="dist-bar-fill ${cls}" data-pct="${pct}"></div>
      </div>
      <span class="dist-pct">${pct}%</span>
    `;

    container.appendChild(row);
  });

  // Animate bars after a paint frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".dist-bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.pct + "%";
      });
    });
  });
}

/**
 * Render per-chunk sentiment items.
 */
function renderChunkList(chunks) {
  const list = $("chunk-list");
  list.innerHTML = "";

  if (!chunks.length) {
    list.innerHTML = '<p style="color:var(--text-muted);font-size:0.83rem;padding:0.5rem;">No chunk data available.</p>';
    return;
  }

  const iconMap = { Positive: "", Neutral: "", Negative: "" };

  chunks.forEach((chunk, idx) => {
    const item = document.createElement("div");
    item.className = "chunk-item";
    item.style.animationDelay = `${idx * 40}ms`;

    const senti    = chunk.sentiment || "Neutral";
    const cls      = senti.toLowerCase();
    const icon     = iconMap[senti] || "";
    const summary  = chunk.summary || chunk.explanation || "No summary available.";
    const words    = chunk.word_count ? `${chunk.word_count.toLocaleString()} words` : "";

    item.innerHTML = `
      <span class="chunk-num">#${chunk.chunk_index || (idx + 1)}</span>
      <span class="chunk-sentiment-pill ${cls}">${icon} ${senti}</span>
      <span class="chunk-summary" title="${escapeHtml(summary)}">${escapeHtml(summary)} <span style="color:var(--text-muted);font-size:0.68rem;">${words}</span></span>
    `;

    list.appendChild(item);
  });
}

/*  UI State Helpers  */
function showLoading() {
  $("empty-state").style.display   = "none";
  $("loading-state").style.display = "flex";
  $("results-content").style.display = "none";
}

function showEmpty() {
  $("empty-state").style.display   = "flex";
  $("loading-state").style.display = "none";
  $("results-content").style.display = "none";
  loadingTimers.forEach(clearTimeout);
}

function showResults() {
  $("empty-state").style.display   = "none";
  $("loading-state").style.display = "none";
  $("results-content").style.display = "flex";
  loadingTimers.forEach(clearTimeout);
}

function setAnalyzeBtn(disabled) {
  const btn = $("analyze-btn");
  btn.disabled = disabled;
  $("btn-text").textContent = disabled ? "Analyzing" : "Analyze Filing";
}

function smoothScrollTo(id) {
  const el = $(id);
  if (!el) return;
  const y = el.getBoundingClientRect().top + window.scrollY - 90;
  window.scrollTo({ top: y, behavior: "smooth" });
}

/*  Toast Notifications  */
let toastTimeout;

function showToast(message, type = "info") {
  // Remove existing toast
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  clearTimeout(toastTimeout);

  const colorMap = {
    info:    "var(--accent-primary)",
    success: "var(--accent-green)",
    error:   "var(--accent-red)",
  };

  const iconMap = {
    info:    "fa-circle-info",
    success: "fa-circle-check",
    error:   "fa-circle-xmark",
  };

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `
    <i class="fas ${iconMap[type] || "fa-circle-info"}"></i>
    <span>${message}</span>
  `;

  Object.assign(toast.style, {
    position: "fixed",
    bottom: "1.5rem",
    right: "1.5rem",
    zIndex: "10000",
    display: "flex",
    alignItems: "center",
    gap: "0.6rem",
    padding: "0.75rem 1.1rem",
    background: "var(--bg-card)",
    border: `1px solid ${colorMap[type]}`,
    borderRadius: "10px",
    color: colorMap[type],
    fontSize: "0.85rem",
    fontWeight: "500",
    boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
    animation: "fade-up 0.3s ease both",
    maxWidth: "360px",
    wordBreak: "break-word",
  });

  document.body.appendChild(toast);

  toastTimeout = setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

/*  Terminal Typing Animation  */
function initTerminalTyping() {
  const lines = document.querySelectorAll(".term-line");
  lines.forEach((line, i) => {
    line.style.opacity = "0";
    line.style.transform = "translateX(-6px)";
    line.style.transition = "opacity 0.3s ease, transform 0.3s ease";
    setTimeout(() => {
      line.style.opacity = "1";
      line.style.transform = "translateX(0)";
    }, 400 + i * 280);
  });
}

/*  Utilities  */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
