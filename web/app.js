/**
 * SMS Spam Shield - Modern Vanilla JS Client
 */

// Presets dictionary
const PRESETS = {
  prize: "WINNER!! You have been selected for a £1,000 cash reward. Text CLAIM to 87007 or visit http://win-now.com to collect your voucher.",
  bank: "URGENT! We detected suspicious activity on your bank account. Verify your identity immediately at http://secure-login.bit.ly",
  delivery: "Your package 849203 is arriving today. Confirm delivery address and pay £1.50 fee at http://track-pkg.info",
  casual: "Hey are we still meeting up for coffee at 6pm today? Let me know!",
  meeting: "Can you please email me the lecture slides from this morning whenever you get home?"
};

// DOM Elements
const messageInput = document.getElementById("messageInput");
const charCount = document.getElementById("charCount");
const clearBtn = document.getElementById("clearBtn");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdVal = document.getElementById("thresholdVal");
const thresholdBadge = document.getElementById("thresholdBadge");
const analyzeBtn = document.getElementById("analyzeBtn");
const verdictBanner = document.getElementById("verdictBanner");
const verdictIcon = document.getElementById("verdictIcon");
const verdictHeadline = document.getElementById("verdictHeadline");
const verdictDesc = document.getElementById("verdictDesc");
const verdictLatency = document.getElementById("verdictLatency");
const probPercentage = document.getElementById("probPercentage");
const meterFill = document.getElementById("meterFill");
const thresholdMarker = document.getElementById("thresholdMarker");
const legendThrText = document.getElementById("legendThrText");
const tokenChips = document.getElementById("tokenChips");
const normText = document.getElementById("normText");
const liveStatus = document.getElementById("liveStatus");

let debounceTimer = null;
let lastAnalyzedText = null;
let lastAnalyzedThreshold = null;

// SVG Icons
const ICON_HAM = `
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    <path d="m9 12 2 2 4-4"/>
  </svg>
`;

const ICON_SPAM = `
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
`;

// Initialize UI
function init() {
  updateCharCount();
  updateThresholdMarker();

  // Attach Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.dataset.preset;
      if (PRESETS[presetKey]) {
        messageInput.value = PRESETS[presetKey];
        updateCharCount();
        analyzeMessage(true);
      }
    });
  });

  // Input Listeners
  messageInput.addEventListener("input", () => {
    updateCharCount();
    triggerDebouncedAnalysis();
  });

  clearBtn.addEventListener("click", () => {
    messageInput.value = "";
    updateCharCount();
    messageInput.focus();
    triggerDebouncedAnalysis();
  });

  // Threshold Slider Listener
  thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    thresholdVal.textContent = val.toFixed(3);
    if (Math.abs(val - 0.709) < 0.005) {
      thresholdBadge.innerHTML = `Calibrated: <strong id="thresholdVal">${val.toFixed(3)}</strong>`;
      thresholdBadge.style.color = "#38bdf8";
    } else if (val < 0.5) {
      thresholdBadge.innerHTML = `Aggressive: <strong id="thresholdVal">${val.toFixed(3)}</strong>`;
      thresholdBadge.style.color = "#fbbf24";
    } else {
      thresholdBadge.innerHTML = `Custom: <strong id="thresholdVal">${val.toFixed(3)}</strong>`;
      thresholdBadge.style.color = "#a78bfa";
    }

    updateThresholdMarker();
    analyzeMessage(false);
  });

  analyzeBtn.addEventListener("click", () => {
    analyzeMessage(true);
  });

  // Initial analysis of default message
  analyzeMessage(true);
}

function updateCharCount() {
  const len = messageInput.value.length;
  charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
}

function updateThresholdMarker() {
  const thr = parseFloat(thresholdSlider.value);
  const percent = Math.min(Math.max(thr * 100, 0), 100);
  thresholdMarker.style.left = `${percent}%`;
  legendThrText.textContent = `Threshold: ${percent.toFixed(1)}%`;
}

function triggerDebouncedAnalysis() {
  liveStatus.textContent = "Typing...";
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    analyzeMessage(false);
  }, 220);
}

async function analyzeMessage(force = false) {
  const text = messageInput.value.trim();
  const threshold = parseFloat(thresholdSlider.value);

  if (!force && text === lastAnalyzedText && threshold === lastAnalyzedThreshold) {
    return;
  }

  lastAnalyzedText = text;
  lastAnalyzedThreshold = threshold;
  liveStatus.textContent = "Scanning...";

  const startTime = performance.now();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, threshold })
    });

    if (!response.ok) {
      throw new Error(`HTTP Error ${response.status}`);
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);
    verdictLatency.textContent = `Latency: ${elapsed}ms`;
    liveStatus.textContent = "Live auto-detection active";

    renderResult(data);
  } catch (err) {
    console.error("Inference Error:", err);
    liveStatus.textContent = "Connection error";
    verdictHeadline.textContent = "Inference Error";
    verdictDesc.textContent = "Could not reach the classification backend.";
  }
}

function renderResult(data) {
  const prob = data.spam_probability || 0.0;
  const percent = (prob * 100).toFixed(1);
  const isSpam = data.is_spam;

  // Percentage & Fill
  probPercentage.textContent = `${percent}%`;
  meterFill.style.width = `${percent}%`;

  if (isSpam) {
    meterFill.className = "meter-fill spam-meter";
    verdictBanner.className = "verdict-banner spam";
    verdictIcon.innerHTML = ICON_SPAM;
    verdictHeadline.textContent = "SPAM DETECTED";
    verdictDesc.textContent = `High risk of unsolicited, fraudulent, or promotional content (p=${prob.toFixed(4)} >= threshold ${data.threshold}).`;
  } else {
    meterFill.className = "meter-fill ham-meter";
    verdictBanner.className = "verdict-banner ham";
    verdictIcon.innerHTML = ICON_HAM;
    verdictHeadline.textContent = "LEGITIMATE (HAM)";
    verdictDesc.textContent = `Message passes verification as safe communication (p=${prob.toFixed(4)} < threshold ${data.threshold}).`;
  }

  // Preprocessing Tokens
  tokenChips.innerHTML = "";
  if (data.extracted_tokens && data.extracted_tokens.length > 0) {
    data.extracted_tokens.forEach(tok => {
      const chip = document.createElement("span");
      chip.className = `token-chip ${tok.type}`;
      chip.textContent = `${tok.token} (${tok.label})`;
      tokenChips.appendChild(chip);
    });
  } else {
    const noTok = document.createElement("span");
    noTok.className = "no-tokens-text";
    noTok.textContent = "No high-risk entity tokens triggered.";
    tokenChips.appendChild(noTok);
  }

  // Normalized Text
  normText.textContent = data.normalized_text || "(empty)";
}

// Start application
document.addEventListener("DOMContentLoaded", init);
