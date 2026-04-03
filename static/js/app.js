/**
 * AI Face Detection Attendance System
 * Frontend JS — WebRTC capture + face-presence timer logic
 *
 * Timer behaviour:
 *   - faceTimer  : counts up only while face IS detected  → pauses on absence
 *   - callTimer  : counts up from session start → always running
 *   - On "End Session": sends both values to /api/session/end
 *                       backend decides PRESENT / ABSENT using threshold
 */

// ── State ─────────────────────────────────────────────────────────────────────
let stream = null;
let captureInterval = null;
let callTimerInterval = null;
let faceTimerInterval = null;

let faceSeconds = 0;       // cumulative face-present seconds (pauses when no face)
let callSeconds = 0;       // total call duration seconds (always runs during session)
let faceDetected = false;  // current frame face status
let sessionActive = false;

const CAPTURE_INTERVAL_MS = 3000;  // how often to send a frame for detection (ms)
const FACE_TICK_MS = 100;          // face timer tick resolution (ms)
const CALL_TICK_MS = 1000;         // call timer tick resolution (ms)

// ── DOM refs (filled in DOMContentLoaded) ─────────────────────────────────────
let video, canvas, ctx, statusBadge, annotatedImg;
let faceTimerEl, callTimerEl, nameInput, thresholdInput;
let startBtn, endBtn, settingsBtn, resultPanel;

// ── Utilities ─────────────────────────────────────────────────────────────────
function formatTime(totalSeconds) {
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = Math.floor(totalSeconds % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatSeconds(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera access denied: " + err.message);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
}

// ── Session ───────────────────────────────────────────────────────────────────
async function startSession() {
    const name = nameInput.value.trim();
    if (!name) { alert("Please enter your name first."); return; }

    // Reset timers
    faceSeconds = 0;
    callSeconds = 0;
    faceDetected = false;
    sessionActive = true;

    updateFaceTimerDisplay();
    updateCallTimerDisplay();
    resultPanel.style.display = "none";
    statusBadge.textContent = "Waiting…";
    statusBadge.className = "status-badge waiting";

    // Notify backend
    await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });

    // Start call-duration timer (always ticking)
    callTimerInterval = setInterval(() => {
        callSeconds += CALL_TICK_MS / 1000;
        updateCallTimerDisplay();
    }, CALL_TICK_MS);

    // Start face-presence timer (only ticks while faceDetected=true)
    faceTimerInterval = setInterval(() => {
        if (faceDetected) {
            faceSeconds += FACE_TICK_MS / 1000;
            updateFaceTimerDisplay();
        }
    }, FACE_TICK_MS);

    // Start periodic frame capture → detection
    captureInterval = setInterval(captureAndDetect, CAPTURE_INTERVAL_MS);
    captureAndDetect(); // immediate first frame

    startBtn.disabled = true;
    endBtn.disabled = false;
    nameInput.disabled = true;
}

async function endSession() {
    if (!sessionActive) return;
    sessionActive = false;

    clearInterval(captureInterval);
    clearInterval(callTimerInterval);
    clearInterval(faceTimerInterval);

    const name = nameInput.value.trim();

    try {
        const res = await fetch("/api/session/end", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name,
                face_seconds: faceSeconds,
                call_duration_seconds: callSeconds,
            }),
        });
        const result = await res.json();
        showResult(result);
    } catch (err) {
        console.error("Session end error:", err);
    }

    startBtn.disabled = false;
    endBtn.disabled = true;
    nameInput.disabled = false;
    faceDetected = false;
    updateStatusBadge(false);
}

// ── Detection ─────────────────────────────────────────────────────────────────
async function captureAndDetect() {
    if (!sessionActive || !stream) return;

    // Draw video frame to canvas
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    const name = nameInput.value.trim() || "Unknown";

    try {
        const res = await fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                frame: dataUrl,
                name,
                face_seconds: faceSeconds,
            }),
        });
        const data = await res.json();
        if (data.error) return;

        faceDetected = data.status === "PRESENT";
        updateStatusBadge(faceDetected);

        if (data.annotated_frame) {
            annotatedImg.src = data.annotated_frame;
        }
    } catch (err) {
        console.error("Detection error:", err);
    }
}

// ── UI updates ────────────────────────────────────────────────────────────────
function updateFaceTimerDisplay() {
    if (faceTimerEl) faceTimerEl.textContent = formatTime(faceSeconds);
}

function updateCallTimerDisplay() {
    if (callTimerEl) callTimerEl.textContent = formatTime(callSeconds);
}

function updateStatusBadge(present) {
    if (!statusBadge) return;
    statusBadge.textContent = present ? "Face Detected ✓" : "No Face ✗";
    statusBadge.className = "status-badge " + (present ? "present" : "absent");
}

function showResult(result) {
    resultPanel.style.display = "block";

    const isPresent = result.final_status === "PRESENT";
    const pct = result.percentage?.toFixed(1) ?? "—";
    const required = formatSeconds(result.required_seconds ?? 0);
    const faceTime = formatSeconds(result.face_seconds ?? 0);
    const callTime = formatSeconds(result.call_duration_seconds ?? 0);
    const thresholdPct = ((result.threshold ?? 0.5) * 100).toFixed(0);

    resultPanel.innerHTML = `
        <div class="result-card ${isPresent ? "result-present" : "result-absent"}">
            <h2>${result.name}</h2>
            <div class="result-status">${result.final_status}</div>
            <table class="result-table">
                <tr><td>Face time</td><td><strong>${faceTime}</strong></td></tr>
                <tr><td>Call duration</td><td><strong>${callTime}</strong></td></tr>
                <tr><td>Required (${thresholdPct}%)</td><td><strong>${required}</strong></td></tr>
                <tr><td>Your presence</td><td><strong>${pct}%</strong></td></tr>
            </table>
            <div class="result-bar-wrap">
                <div class="result-bar" style="width:${Math.min(pct, 100)}%"></div>
                <div class="result-bar-threshold" style="left:${thresholdPct}%"></div>
            </div>
            <p class="result-note">
                ${isPresent
                    ? `✅ Marked <strong>PRESENT</strong> — you met the ${thresholdPct}% threshold.`
                    : `❌ Marked <strong>ABSENT</strong> — face was present for only ${pct}% (needed ${thresholdPct}%).`}
            </p>
        </div>
    `;
}

// ── Config / Settings ─────────────────────────────────────────────────────────
async function loadConfig() {
    try {
        const res = await fetch("/api/config");
        const cfg = await res.json();
        if (thresholdInput) {
            thresholdInput.value = (cfg.presence_threshold * 100).toFixed(0);
        }
    } catch (err) {
        console.error("Config load error:", err);
    }
}

async function saveConfig() {
    const pct = parseFloat(thresholdInput.value);
    if (isNaN(pct) || pct <= 0 || pct > 100) {
        alert("Threshold must be between 1 and 100.");
        return;
    }
    try {
        const res = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ presence_threshold: pct / 100 }),
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }
        alert(`✅ Saved! Attendance threshold set to ${pct}%.`);
        document.getElementById("settings-panel").style.display = "none";
    } catch (err) {
        alert("Failed to save config: " + err.message);
    }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    video       = document.getElementById("video");
    canvas      = document.getElementById("canvas");
    ctx         = canvas.getContext("2d");
    statusBadge = document.getElementById("status-badge");
    annotatedImg = document.getElementById("annotated-frame");
    faceTimerEl = document.getElementById("face-timer");
    callTimerEl = document.getElementById("call-timer");
    nameInput   = document.getElementById("student-name");
    thresholdInput = document.getElementById("threshold-input");
    startBtn    = document.getElementById("start-btn");
    endBtn      = document.getElementById("end-btn");
    settingsBtn = document.getElementById("settings-btn");
    resultPanel = document.getElementById("result-panel");

    startCamera();
    loadConfig();

    startBtn.addEventListener("click", startSession);
    endBtn.addEventListener("click", endSession);

    settingsBtn?.addEventListener("click", () => {
        const panel = document.getElementById("settings-panel");
        panel.style.display = panel.style.display === "none" ? "block" : "none";
    });

    document.getElementById("save-config-btn")?.addEventListener("click", saveConfig);

    // Export button
    document.getElementById("export-btn")?.addEventListener("click", () => {
        window.location.href = "/api/export";
    });

    // Attendance summary
    document.getElementById("refresh-btn")?.addEventListener("click", async () => {
        const res = await fetch("/api/attendance");
        const data = await res.json();
        const el = document.getElementById("attendance-summary");
        if (!el) return;
        el.innerHTML = `
            <p>Date: <strong>${data.date}</strong></p>
            <p>Present: <strong>${data.present}</strong> / ${data.total}</p>
            <p>Absent: <strong>${data.absent}</strong></p>
        `;
    });
});
