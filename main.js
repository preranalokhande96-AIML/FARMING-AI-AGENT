/**
 * Smart Farming Agent — main.js
 * IBM Watsonx.ai · Flask · Granite LLM
 */
"use strict";

// ── State ─────────────────────────────────────────────────────────────────
const State = {
  farmContext: {},
  pendingImage: null,          // base64 image string
  isLoading:   false,
  language:    "English",
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const chatMessages      = document.getElementById("chatMessages");
const chatInput         = document.getElementById("chatInput");
const sendBtn           = document.getElementById("sendBtn");
const clearChatBtn      = document.getElementById("clearChatBtn");
const typingIndicator   = document.getElementById("typingIndicator");
const imageUploadBtn    = document.getElementById("imageUploadBtn");
const imageFileInput    = document.getElementById("imageFileInput");
const imagePreviewBar   = document.getElementById("imagePreviewBar");
const imageThumb        = document.getElementById("imageThumb");
const removeImageBtn    = document.getElementById("removeImageBtn");
const themeToggle       = document.getElementById("themeToggle");
const statusBadge       = document.getElementById("statusBadge");
const suggestedChips    = document.getElementById("suggestedChips");

// Dashboard
const dashCropVal    = document.getElementById("dashCropVal");
const dashSoilVal    = document.getElementById("dashSoilVal");
const dashTempVal    = document.getElementById("dashTempVal");
const dashRainVal    = document.getElementById("dashRainVal");
const dashSeasonVal  = document.getElementById("dashSeasonVal");

// ── Helpers ───────────────────────────────────────────────────────────────
function showToast(msg, type = "success") {
  const toast    = document.getElementById("notifToast");
  const toastMsg = document.getElementById("toastMsg");
  if (!toast || !toastMsg) return;
  toastMsg.textContent = msg;
  toast.className      = `toast align-items-center border-0 text-bg-${type}`;
  bootstrap.Toast.getOrCreateInstance(toast, { delay: 3000 }).show();
}

function getTime() {
  return new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
});

// ── Message Renderer ──────────────────────────────────────────────────────
function renderMarkdown(text) {
  if (window.marked) {
    try {
      return marked.parse(text, { breaks: true, gfm: true });
    } catch (_) { /* fall through */ }
  }
  // Fallback: basic formatting
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g,     "<em>$1</em>")
    .replace(/`(.*?)`/g,       "<code>$1</code>")
    .replace(/\n/g,            "<br/>");
}

function appendMessage(role, content, imageUrl = null) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user-message" : "agent-message"}`;

  const avatarEmoji = role === "user" ? "👨‍🌾" : "🌾";
  const time        = getTime();
  const htmlContent = role === "user"
    ? content.replace(/\n/g, "<br>")
    : renderMarkdown(content);

  wrapper.innerHTML = `
    <div class="message-avatar">${avatarEmoji}</div>
    <div class="message-body">
      ${imageUrl ? `<img src="${imageUrl}" class="img-fluid rounded mb-2" style="max-height:180px;max-width:260px;object-fit:cover" />` : ""}
      <div class="message-content">${htmlContent}</div>
      <div class="message-time">${time}</div>
    </div>`;

  chatMessages.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function scrollToBottom() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

// ── Typing Indicator ──────────────────────────────────────────────────────
function showTyping()  {
  typingIndicator.style.display = "block";
  chatMessages.appendChild(typingIndicator);
  scrollToBottom();
}

function hideTyping()  {
  typingIndicator.style.display = "none";
  document.getElementById("chatMessages").appendChild;   // no-op safety
}

// ── Core Chat Send ────────────────────────────────────────────────────────
async function sendMessage() {
  if (State.isLoading) return;
  const text  = chatInput.value.trim();
  const image = State.pendingImage;

  if (!text && !image) return;

  // Render user message
  appendMessage("user", text || "(Image sent)", image);

  // Reset input
  chatInput.value         = "";
  chatInput.style.height  = "auto";
  clearImageAttachment();

  // Hide chips after first message
  if (suggestedChips) suggestedChips.style.display = "none";

  State.isLoading = true;
  sendBtn.disabled = true;
  showTyping();

  try {
    const payload = {
      message:      text,
      farm_context: State.farmContext,
    };
    if (image) payload.image = image;

    const response = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();
    hideTyping();

    if (data.error) {
      appendMessage("agent", `⚠️ Error: ${data.error}`);
    } else {
      appendMessage("agent", data.response);
      updateStatusBadge(data.mode);
    }
  } catch (err) {
    hideTyping();
    appendMessage("agent", "⚠️ Network error. Please check your connection and try again.");
    console.error("Chat error:", err);
  } finally {
    State.isLoading  = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

// ── Status Badge ──────────────────────────────────────────────────────────
function updateStatusBadge(mode) {
  if (!statusBadge) return;
  if (mode === "live") {
    statusBadge.className   = "badge status-badge-live";
    statusBadge.innerHTML   = `<i class="bi bi-circle-fill me-1" style="font-size:0.55rem"></i>Live`;
  } else {
    statusBadge.className   = "badge status-badge-demo";
    statusBadge.innerHTML   = `<i class="bi bi-circle-fill me-1" style="font-size:0.55rem"></i>Demo`;
  }
}

// ── Send Button / Enter Key ───────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Clear Chat ────────────────────────────────────────────────────────────
clearChatBtn.addEventListener("click", async () => {
  if (!confirm("Clear conversation history?")) return;
  await fetch("/api/clear-chat", { method: "POST" });

  // Keep only the welcome message (first child)
  const msgs = chatMessages.querySelectorAll(".message");
  msgs.forEach((m, i) => { if (i > 0) m.remove(); });

  if (suggestedChips) suggestedChips.style.display = "flex";
  showToast("Chat cleared ✓", "success");
});

// ── Image Upload ──────────────────────────────────────────────────────────
imageUploadBtn.addEventListener("click", () => imageFileInput.click());

imageFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showToast("Please select a valid image file.", "warning");
    return;
  }

  const reader = new FileReader();
  reader.onload = (ev) => {
    State.pendingImage    = ev.target.result;
    imageThumb.src        = ev.target.result;
    imagePreviewBar.style.display = "block";
  };
  reader.readAsDataURL(file);
  imageFileInput.value = "";  // reset so same file can be re-selected
});

removeImageBtn.addEventListener("click", clearImageAttachment);

function clearImageAttachment() {
  State.pendingImage           = null;
  imagePreviewBar.style.display = "none";
  imageThumb.src               = "";
}

// ── Suggestion Chips ──────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    chatInput.value = chip.dataset.msg;
    sendMessage();
  });
});

// ── Quick Advisory Buttons ────────────────────────────────────────────────
document.querySelectorAll(".quick-btn").forEach((btn) => {
  btn.addEventListener("click", async function () {
    const category = this.dataset.category;
    const crop     = getCombinedValue("currentCrop", "mCurrentCrop");

    if (!crop) {
      showToast("Please select a crop in your Farm Profile first.", "warning");
      return;
    }

    this.classList.add("loading");
    const originalHTML = this.innerHTML;
    this.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Loading…`;

    // Add user bubble
    const label = this.textContent.trim();
    appendMessage("user", `📋 Quick Advisory: ${label} for ${crop}`);
    if (suggestedChips) suggestedChips.style.display = "none";
    showTyping();

    try {
      const res  = await fetch("/api/quick-advice", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ category, crop, context: State.farmContext }),
      });
      const data = await res.json();
      hideTyping();
      appendMessage("agent", data.response);
      updateStatusBadge(data.mode || "demo");
    } catch (err) {
      hideTyping();
      appendMessage("agent", "⚠️ Could not fetch advisory. Please try again.");
    } finally {
      this.classList.remove("loading");
      this.innerHTML = originalHTML;
    }
  });
});

// ── Farm Profile Save ─────────────────────────────────────────────────────
function getVal(id)           { return document.getElementById(id)?.value?.trim() || ""; }
function getCombinedValue(...ids) {
  for (const id of ids) {
    const v = getVal(id);
    if (v) return v;
  }
  return "";
}

async function saveProfile() {
  const name     = getCombinedValue("farmerName",    "mFarmerName");
  const location = getCombinedValue("farmerLocation","mFarmerLocation");
  const crop     = getCombinedValue("currentCrop",   "mCurrentCrop");
  const season   = getCombinedValue("farmSeason",    "mFarmSeason");

  const ctx = {};
  if (name)     ctx.farmer_name = name;
  if (location) ctx.location    = location;
  if (crop)     ctx.crop        = crop;
  if (season)   ctx.season      = season;

  Object.assign(State.farmContext, ctx);
  await sendContextToServer(ctx);

  // Update dashboard
  if (crop)    { dashCropVal.textContent   = titleCase(crop); }
  if (season)  { dashSeasonVal.textContent = titleCase(season); }

  // Sync mobile↔desktop
  syncInputs("currentCrop",    "mCurrentCrop",    crop);
  syncInputs("farmSeason",     "mFarmSeason",     season);
  syncInputs("farmerName",     "mFarmerName",     name);
  syncInputs("farmerLocation", "mFarmerLocation", location);

  showToast("✅ Farm profile saved!", "success");

  // Contextual greeting
  if (name && crop) {
    appendMessage("agent",
      `👋 Namaste **${name}**! Your farm profile is updated.\n\n` +
      `I see you're growing **${crop}** this **${season || "season"}** in **${location || "your region"}**.\n\n` +
      `How can I help you today? 🌾`
    );
    if (suggestedChips) suggestedChips.style.display = "none";
  }
}

async function saveSoilWeather() {
  const soilType    = getCombinedValue("soilType",    "mSoilType");
  const soilPh      = getCombinedValue("soilPh",      "mSoilPh");
  const temperature = getCombinedValue("temperature", "mTemperature");
  const rainfall    = getCombinedValue("rainfall",    "mRainfall");

  if (!soilType && !soilPh && !temperature && !rainfall) {
    showToast("Please fill in at least one soil or weather field.", "warning");
    return;
  }

  const ctx = {};
  if (soilType)    ctx.soil_type = soilType;
  if (soilPh)      ctx.soil_ph   = soilPh;
  if (temperature) ctx.weather   = `${temperature}°C`;
  if (rainfall)    ctx.rainfall  = `${rainfall}mm`;

  Object.assign(State.farmContext, ctx);
  await sendContextToServer(ctx);

  // ── Update dashboard (fixed: always update, use "—" if cleared) ──────────
  dashSoilVal.textContent = soilType    ? titleCase(soilType)   : (State.farmContext.soil_type ? titleCase(State.farmContext.soil_type) : "—");
  dashTempVal.textContent = temperature ? `${temperature}°C`    : (State.farmContext.weather   ? State.farmContext.weather             : "—");
  dashRainVal.textContent = rainfall    ? `${rainfall}mm`       : (State.farmContext.rainfall  ? State.farmContext.rainfall            : "—");

  // Sync mobile↔desktop fields
  syncInputs("soilType",    "mSoilType",    soilType);
  syncInputs("soilPh",      "mSoilPh",      soilPh);
  syncInputs("temperature", "mTemperature", temperature);
  syncInputs("rainfall",    "mRainfall",    rainfall);

  showToast("🌱 Soil & weather context updated!", "success");

  // Give a contextual response in the chat
  const parts = [];
  if (soilType)    parts.push(`soil: **${titleCase(soilType)}**`);
  if (soilPh)      parts.push(`pH: **${soilPh}**`);
  if (temperature) parts.push(`temperature: **${temperature}°C**`);
  if (rainfall)    parts.push(`rainfall: **${rainfall}mm**`);
  if (parts.length) {
    appendMessage("agent",
      `🌍 Farm context updated — I've noted your ${parts.join(", ")}.\n\n` +
      `This will now be used to personalise all my recommendations for your **${State.farmContext.crop || "crop"}**. 🌾`
    );
    if (suggestedChips) suggestedChips.style.display = "none";
  }
}

async function sendContextToServer(ctx) {
  try {
    await fetch("/api/update-farm-context", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(ctx),
    });
  } catch (_) { /* non-critical */ }
}

// Wire save buttons (desktop + mobile)
document.getElementById("saveProfileBtn")?.addEventListener("click", saveProfile);
document.getElementById("mSaveProfileBtn")?.addEventListener("click", saveProfile);
document.getElementById("saveSoilBtn")?.addEventListener("click", saveSoilWeather);
document.getElementById("mSaveSoilBtn")?.addEventListener("click", saveSoilWeather);

// ── Language Switcher ─────────────────────────────────────────────────────
document.querySelectorAll(".lang-option").forEach((opt) => {
  opt.addEventListener("click", async (e) => {
    e.preventDefault();
    const lang = opt.dataset.lang;
    State.language = lang;
    const labels = { English: "EN", Hindi: "HI", Marathi: "MR" };
    document.getElementById("currentLang").textContent = labels[lang] || "EN";

    await sendContextToServer({ preferred_language: lang });

    const msgs = {
      English: "🌐 I'll respond in **English** from now on.",
      Hindi:   "🌐 अब मैं **हिंदी** में जवाब दूंगा।",
      Marathi: "🌐 आता मी **मराठी** मध्ये उत्तर देईन.",
    };
    appendMessage("agent", msgs[lang] || msgs.English);
    showToast(`Language set to ${lang}`, "success");
  });
});

// ── Dark Mode Toggle ──────────────────────────────────────────────────────
const html       = document.documentElement;
const savedTheme = localStorage.getItem("sfa-theme") || "light";
html.setAttribute("data-theme", savedTheme);
updateThemeIcon(savedTheme);

themeToggle.addEventListener("click", () => {
  const current = html.getAttribute("data-theme");
  const next    = current === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  localStorage.setItem("sfa-theme", next);
  updateThemeIcon(next);
});

function updateThemeIcon(theme) {
  themeToggle.innerHTML = theme === "dark"
    ? `<i class="bi bi-sun-fill"></i>`
    : `<i class="bi bi-moon-stars-fill"></i>`;
}

// ── Voice Input (Web Speech API) ──────────────────────────────────────────
const voiceBtn = document.getElementById("voiceBtn");
let recognition = null;

if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition             = new SpeechRecognition();
  recognition.lang        = "hi-IN";    // Hindi, works for English too
  recognition.continuous  = false;
  recognition.interimResults = false;

  recognition.onresult = (e) => {
    const transcript      = e.results[0][0].transcript;
    chatInput.value       = transcript;
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
    voiceBtn.innerHTML    = `<i class="bi bi-mic-fill"></i>`;
    voiceBtn.style.color  = "";
    showToast(`🎙️ Heard: "${transcript}"`, "info");
  };

  recognition.onerror = () => {
    voiceBtn.innerHTML   = `<i class="bi bi-mic-fill"></i>`;
    voiceBtn.style.color = "";
  };

  voiceBtn.addEventListener("click", () => {
    recognition.start();
    voiceBtn.innerHTML   = `<i class="bi bi-mic-mute-fill"></i>`;
    voiceBtn.style.color = "var(--primary)";
    showToast("🎙️ Listening… Speak now (Hindi/English/Marathi)", "info");
  });
} else {
  voiceBtn.title   = "Voice input not supported in this browser";
  voiceBtn.opacity = 0.4;
}

// ── Utility ───────────────────────────────────────────────────────────────
function titleCase(str) {
  if (!str) return "";
  return str.replace(/\w\S*/g, (t) => t.charAt(0).toUpperCase() + t.slice(1).toLowerCase());
}

function syncInputs(desktopId, mobileId, value) {
  const d = document.getElementById(desktopId);
  const m = document.getElementById(mobileId);
  if (d && value !== undefined) d.value = value;
  if (m && value !== undefined) m.value = value;
}

// ── Init: Check Watsonx Status ────────────────────────────────────────────
(async function checkStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();
    updateStatusBadge(data.mode);
  } catch (_) { /* server not ready yet */ }
})();

// ── Setup Guide Modal — live checklist ────────────────────────────────────
const setupModal = document.getElementById("setupModal");
if (setupModal) {
  setupModal.addEventListener("show.bs.modal", loadSetupChecks);
}

async function loadSetupChecks() {
  const spinner    = document.getElementById("setupSpinner");
  const statusMsg  = document.getElementById("setupStatusMsg");
  const nextStep   = document.getElementById("setupNextStep");
  const nextText   = document.getElementById("setupNextStepText");
  const errorBox   = document.getElementById("setupErrorBox");
  const errorText  = document.getElementById("setupErrorText");

  if (spinner)   { spinner.style.display = "inline-block"; }
  if (statusMsg) { statusMsg.textContent = "Checking configuration…"; statusMsg.style.color = ""; }

  // Reset all check items to pending state
  document.querySelectorAll(".check-item").forEach((el) => {
    el.classList.remove("check-pass", "check-fail");
    el.querySelector(".check-icon").textContent = "⏳";
  });
  if (nextStep)  nextStep.style.display  = "none";
  if (errorBox)  errorBox.style.display  = "none";

  try {
    const res  = await fetch("/api/setup-check");
    const data = await res.json();

    // ── Update each check item ────────────────────────────────────────────
    const checkMap = {
      env_file_loaded: "chk-env_file_loaded",
      api_key_set:     "chk-api_key_set",
      project_id_set:  "chk-project_id_set",
      sdk_installed:   "chk-sdk_installed",
      model_connected: "chk-model_connected",
    };

    for (const [key, elemId] of Object.entries(checkMap)) {
      const el   = document.getElementById(elemId);
      const icon = el?.querySelector(".check-icon");
      if (!el) continue;
      const passed = data.checks?.[key];
      el.classList.remove("check-pass", "check-fail");
      el.classList.add(passed ? "check-pass" : "check-fail");
      if (icon) icon.textContent = passed ? "✅" : "❌";

      // Annotate model_connected with which model is active
      if (key === "model_connected" && passed && data.model_used) {
        const desc = el.querySelector(".check-desc");
        if (desc) desc.innerHTML = `Connected: <span class="model-tag">${data.model_used}</span>`;
      }
    }

    // ── Status bar ────────────────────────────────────────────────────────
    if (spinner) spinner.style.display = "none";
    if (statusMsg) {
      statusMsg.textContent = data.all_ok
        ? `✅ Live — ${data.model_used || "Granite"} connected`
        : `⚠️ ${data.missing?.length || 0} item(s) need attention — see below`;
      statusMsg.style.color = data.all_ok ? "var(--primary)" : "#f59e0b";
    }

    // ── Next step hint ────────────────────────────────────────────────────
    if (nextStep && nextText && data.next_step && !data.error_detail) {
      nextText.textContent   = data.next_step;
      nextStep.style.display = "flex";
    }

    // ── Error detail (model_connected fail) ───────────────────────────────
    if (errorBox && errorText && data.error_detail) {
      errorText.textContent  = data.error_detail;
      errorBox.style.display = "block";
    }

    // ── Refresh navbar badge ──────────────────────────────────────────────
    updateStatusBadge(data.mode || "demo");

  } catch (err) {
    if (spinner)   spinner.style.display  = "none";
    if (statusMsg) statusMsg.textContent  = "⚠️ Could not reach server — is Flask running?";
    console.error("setup-check error:", err);
  }
}

// ── Test Connection button (runs /api/diagnose) ───────────────────────────
document.getElementById("testConnectionBtn")?.addEventListener("click", runDiagnose);

async function runDiagnose() {
  const btn    = document.getElementById("testConnectionBtn");
  const panel  = document.getElementById("diagnosePanel");
  const output = document.getElementById("diagnoseOutput");

  if (!panel || !output) return;

  btn?.classList.add("testing");
  if (btn) { btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span>Testing…`; }

  panel.style.display  = "block";
  output.innerHTML     = `<span class="diagnose-line-info">▶ Running live connection test…</span>\n`;

  try {
    const res  = await fetch("/api/diagnose");
    const data = await res.json();

    let html = "";

    const line = (cls, txt) => `<span class="${cls}">${escHtml(txt)}</span>\n`;

    html += line("diagnose-line-info", `SDK available:      ${data.sdk_available}`);
    html += line("diagnose-line-info", `API key prefix:     ${data.api_key_prefix}`);
    html += line("diagnose-line-info", `Project ID prefix:  ${data.project_id_prefix}`);
    html += line("diagnose-line-info", `Watsonx URL:        ${data.watsonx_url}`);
    html += "\n";

    for (const entry of (data.models_tried || [])) {
      if (entry.status === "success") {
        html += line("diagnose-line-ok",   `✅ ${entry.model_id} — OK`);
        if (entry.test_output) html += line("diagnose-line-ok", `   Output: ${entry.test_output}`);
      } else {
        html += line("diagnose-line-fail", `❌ ${entry.model_id} — ${entry.error}`);
      }
    }

    html += "\n";
    if (data.connected) {
      html += line("diagnose-line-ok",   `✅ CONNECTED — Model: ${data.model_used}`);
      // Refresh the checklist
      await loadSetupChecks();
    } else {
      html += line("diagnose-line-fail", `❌ NOT CONNECTED`);
      html += line("diagnose-line-warn", `   ${data.error || "Unknown error"}`);
      if (data.raw_error) {
        html += line("diagnose-line-fail", `\nRaw error:\n${data.raw_error.substring(0, 400)}`);
      }
    }

    output.innerHTML += html;

  } catch (err) {
    output.innerHTML += `<span class="diagnose-line-fail">Network error: ${escHtml(String(err))}</span>`;
  } finally {
    btn?.classList.remove("testing");
    if (btn) btn.innerHTML = `<i class="bi bi-wifi me-1"></i>Test Connection`;
    output.scrollTop = output.scrollHeight;
  }
}

document.getElementById("closeDiagnoseBtn")?.addEventListener("click", () => {
  const panel = document.getElementById("diagnosePanel");
  if (panel) panel.style.display = "none";
});

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
