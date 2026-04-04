/**
 * AI-Based Face Detection Attendance System
 * Frontend: WebRTC + Fetch API
 *
 * SESSION LOGIC:
 * - Student picks a duration via the session buttons, enters name → clicks Start Session
 * - /api/start-session is called → session registered on backend
 * - Countdown timer (SVG ring) runs for the FULL duration, independent of face detection
 * - Frames are captured & sent to /api/detect every CAPTURE_INTERVAL_MS throughout the session
 * - When timer hits 0 (or End Session Early clicked) → /api/end-session is called
 * - Backend applies the 75% rule and returns PRESENT / ABSENT
 * - Result card is shown; attendance table is refreshed
 */

// ── Config ────────────────────────────────────────────────────────────────────
const CAPTURE_INTERVAL_MS = 3000; // capture a frame every 3 seconds

// ── State ─────────────────────────────────────────────────────────────────────
let stream             = null;
let sessionId          = null;
let sessionDurationSec = 0;   // total session length chosen by user
let sessionElapsedSec  = 0;   // seconds elapsed since session start
let selectedSeconds    = 0;   // from the session-btn buttons

let captureIntervalId   = null; // setInterval handle for frame capture
let countdownIntervalId = null; // setInterval handle for 1-sec countdown

// SVG ring circumference (r=52 → C = 2πr ≈ 326.7)
const RING_CIRC = 2 * Math.PI * 52;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video         = document.getElementById("video");
const overlayCanvas = document.getElementById("overlay");       // canvas overlaid on video
const hiddenCanvas  = document.createElement("canvas");         // off-screen canvas for capture

const startBtn      = document.getElementById("startBtn");
const endBtn        = document.getElementById("endBtn");
const newSessionBtn = document.getElementById("newSessionBtn");
const refreshBtn    = document.getElementById("refreshBtn");
const exportBtn     = document.getElementById("exportBtn");

const nameInput     = document.getElementById("studentName");
const sessionBtns   = document.querySelectorAll(".session-btn");

// Cards
const setupCard     = document.getElementById("setupCard");
const sessionCard   = document.getElementById("sessionCard");
const resultCard    = document.getElementById("resultCard");
const cameraIdle    = document.getElementById("cameraIdle");
const recDot        = document.getElementById("recDot");

// Timer ring
const timerDisplay  = document.getElementById("timerDisplay");
const ringProgress  = document.getElementById("ringProgress");

// Presence bar
const presencePct   = document.getElementById("presencePct");
const presenceFill  = document.getElementById("presenceFill");

// Status pill
const statusPill    = document.getElementById("statusPill");
const pillDot       = document.getElementById("pillDot");
const pillText      = document.getElementById("pillText");

// Result card elements
const resultBadge    = document.getElementById("resultBadge");
const resFaceTime    = document.getElementById("resFaceTime");
const resSessionTime = document.getElementById("resSessionTime");
const resPct         = document.getElementById("resPct");

// Attendance table
const attBody        = document.getElementById("attBody");
const attSummary     = document.getElementById("attSummary");

// ── Utility ───────────────────────────────────────────────────────────────────

function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function formatTime(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = (sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

/** Convert total seconds → readable "Xm Ys" string */
function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

// ── SVG ring update ───────────────────────────────────────────────────────────

function updateRing(elapsed, total) {
  const fraction = total > 0 ? elapsed / total : 0;
  // Ring fills clockwise as time elapses
  const offset = RING_CIRC * (1 - fraction);
  ringProgress.style.strokeDashoffset = offset;
}

// ── Presence bar update ───────────────────────────────────────────────────────

function updatePresenceBar(pct) {
  const clamped = Math.min(Math.max(pct, 0), 100);
  presenceFill.style.width      = `${clamped}%`;
  presencePct.textContent       = `${clamped.toFixed(1)}%`;
  // Green above threshold, red below
  presenceFill.style.background = clamped >= 75 ? "#22c55e" : "#ef4444";
}

// ── Status pill update ────────────────────────────────────────────────────────

function setPill(found) {
  if (found) {
    pillDot.style.background    = "#22c55e";
    pillText.textContent        = "Face Detected";
    statusPill.style.borderColor = "#22c55e33";
  } else {
    pillDot.style.background    = "#ef4444";
    pillText.textContent        = "No Face Detected";
    statusPill.style.borderColor = "#ef444433";
  }
}

// ── Camera ────────────────────────────────────────────────────────────────────

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
  hide(cameraIdle);
  if (recDot) recDot.classList.add("active");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  show(cameraIdle);
  if (recDot) recDot.classList.remove("active");

  // Clear overlay canvas
  if (overlayCanvas) {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }
}

// ── Frame capture & detection ─────────────────────────────────────────────────

function captureFrame() {
  hiddenCanvas.width  = video.videoWidth  || 640;
  hiddenCanvas.height = video.videoHeight || 480;
  hiddenCanvas.getContext("2d").drawImage(video, 0, 0);
  return hiddenCanvas.toDataURL("image/jpeg", 0.8);
}

/** Draw the annotated frame returned by the backend onto the overlay canvas */
function drawAnnotated(dataUrl) {
  const img  = new Image();
  img.onload = () => {
    overlayCanvas.width  = img.width;
    overlayCanvas.height = img.height;
    overlayCanvas.getContext("2d").drawImage(img, 0, 0);
  };
  img.src = dataUrl;
}

async function sendFrame() {
  // Guard: only send if a session is active
  if (!sessionId) return;

  let frameDataUrl;
  try {
    frameDataUrl = captureFrame();
  } catch (e) {
    console.warn("captureFrame error:", e);
    return;
  }

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frameDataUrl, session_id: sessionId }),
    });

    if (!res.ok) { console.warn("Detect API error:", res.status); return; }

    const data = await res.json();

    // Draw annotated bounding boxes on the overlay canvas
    if (data.annotated_frame) drawAnnotated(data.annotated_frame);

    // Update the status pill (face found or not)
    setPill(data.face_found);

    // Update the presence bar with the running percentage from the backend
    if (data.session_stats) {
      updatePresenceBar(data.session_stats.presence_pct);
    }

  } catch (err) {
    console.error("sendFrame error:", err);
  }
}

// ── Attendance table ──────────────────────────────────────────────────────────

async function loadAttendance() {
  try {
    const res  = await fetch("/api/attendance");
    if (!res.ok) return;
    const data = await res.json();

    const records = data.records || {};
    const keys    = Object.keys(records);

    if (keys.length === 0) {
      attBody.innerHTML     = `<tr class="empty-row"><td colspan="6">No records yet</td></tr>`;
      attSummary.textContent = "";
      return;
    }

    attBody.innerHTML = keys.map(name => {
      const r         = records[name];
      const isPresent = r.status === "PRESENT";
      // Approximate face time and session time from frame counts × capture interval
      const faceTimeSec    = Math.round(r.face_frames    * (CAPTURE_INTERVAL_MS / 1000));
      const sessionTimeSec = Math.round(r.total_frames   * (CAPTURE_INTERVAL_MS / 1000));
      const endTime        = r.session_end ? r.session_end.slice(11, 19) : "—";

      return `
        <tr>
          <td>${name}</td>
          <td>
            <span class="status-tag ${isPresent ? "present" : "absent"}">
              ${isPresent ? "✅ Present" : "❌ Absent"}
            </span>
          </td>
          <td>${formatDuration(faceTimeSec)}</td>
          <td>${formatDuration(sessionTimeSec)}</td>
          <td>${parseFloat(r.presence_pct).toFixed(1)}%</td>
          <td>${endTime}</td>
        </tr>`;
    }).join("");

    attSummary.textContent =
      `Total: ${data.total}  |  Present: ${data.present}  |  Absent: ${data.absent}`;

  } catch (err) {
    console.error("loadAttendance error:", err);
  }
}

// ── Session lifecycle ─────────────────────────────────────────────────────────

async function startSession() {
  const name = nameInput.value.trim();

  if (!name) {
    alert("Please enter your name.");
    return;
  }
  if (!selectedSeconds || selectedSeconds <= 0) {
    alert("Please select a session duration.");
    return;
  }

  sessionDurationSec = selectedSeconds;
  sessionElapsedSec  = 0;
  sessionId          = generateSessionId();

  // 1. Start the camera
  try {
    await startCamera();
  } catch (err) {
    alert("Camera access denied: " + err.message);
    return;
  }

  // 2. Register session on backend
  try {
    const res = await fetch("/api/start-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, session_id: sessionId }),
    });
    if (!res.ok) throw new Error("start-session API failed");
  } catch (err) {
    alert("Could not start session on server: " + err.message);
    stopCamera();
    return;
  }

  // 3. Switch UI: hide setup & result, show session card
  hide(setupCard);
  hide(resultCard);
  show(sessionCard);

  // 4. Reset ring, bar, pill to initial state
  updateRing(0, sessionDurationSec);
  updatePresenceBar(0);
  timerDisplay.textContent = formatTime(sessionDurationSec);
  setPill(false);

  // 5. ── FRAME CAPTURE LOOP ──────────────────────────────────────────────────
  //    Runs every CAPTURE_INTERVAL_MS for the ENTIRE session.
  //    Does NOT stop when face is absent. Does NOT start only when face is found.
  //    This is the core fix — capturing is completely decoupled from detection result.
  captureIntervalId = setInterval(sendFrame, CAPTURE_INTERVAL_MS);

  // 6. ── COUNTDOWN LOOP ─────────────────────────────────────────────────────
  //    Runs every 1 second. Completely independent of face detection.
  //    Updates the SVG ring and timer display regardless of what the camera sees.
  countdownIntervalId = setInterval(() => {
    sessionElapsedSec += 1;
    const remaining = sessionDurationSec - sessionElapsedSec;

    timerDisplay.textContent = formatTime(Math.max(remaining, 0));
    updateRing(sessionElapsedSec, sessionDurationSec);

    if (remaining <= 0) {
      // Session over — finalize attendance
      finalizeSession();
    }
  }, 1000);
}

async function finalizeSession() {
  // Stop both loops immediately
  clearInterval(captureIntervalId);
  clearInterval(countdownIntervalId);
  captureIntervalId   = null;
  countdownIntervalId = null;

  stopCamera();

  const sid = sessionId;
  sessionId = null; // clear early to prevent double-calls

  if (!sid) return;

  // Call backend — applies 75% rule and writes to CSV
  try {
    const res = await fetch("/api/end-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    });

    if (!res.ok) throw new Error("end-session API failed");
    const data = await res.json();

    // ── Populate result card ───────────────────────────────────────────────
    const isPresent = data.status === "PRESENT";

    resultBadge.textContent = isPresent ? "✅ PRESENT" : "❌ ABSENT";
    resultBadge.className   = `result-badge ${isPresent ? "present" : "absent"}`;

    // Approximate durations from frame counts × capture interval
    const faceTimeSec    = data.face_frames  * (CAPTURE_INTERVAL_MS / 1000);
    const sessionTimeSec = data.total_frames * (CAPTURE_INTERVAL_MS / 1000);

    resFaceTime.textContent    = formatDuration(Math.round(faceTimeSec));
    resSessionTime.textContent = formatDuration(Math.round(sessionTimeSec));
    resPct.textContent         = `${data.presence_pct}%`;

    // Switch to result card
    hide(sessionCard);
    show(resultCard);

    // Refresh attendance table with the new record
    loadAttendance();

  } catch (err) {
    console.error("finalizeSession error:", err);
    alert("Error finalizing session. Check the console.");
    hide(sessionCard);
    show(setupCard);
  }
}

function resetToSetup() {
  hide(resultCard);
  hide(sessionCard);
  show(setupCard);

  // Deselect all session duration buttons
  sessionBtns.forEach(b => b.classList.remove("active"));
  selectedSeconds   = 0;
  startBtn.disabled = true;

  // Reset ring and bar visually
  timerDisplay.textContent = "00:00";
  updateRing(0, 1);
  updatePresenceBar(0);
  setPill(false);
}

// ── Session button selection ──────────────────────────────────────────────────

sessionBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    // Highlight selected button
    sessionBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedSeconds = parseInt(btn.dataset.seconds, 10);

    // Enable Start Session only when both name and duration are set
    startBtn.disabled = !nameInput.value.trim();
  });
});

// Enable / disable Start Session as name input changes
nameInput.addEventListener("input", () => {
  startBtn.disabled = !(nameInput.value.trim() && selectedSeconds > 0);
});

// ── Button event listeners ────────────────────────────────────────────────────

startBtn.addEventListener("click",      startSession);
endBtn.addEventListener("click",        finalizeSession);    // manual early end
newSessionBtn.addEventListener("click", resetToSetup);
refreshBtn.addEventListener("click",    loadAttendance);
exportBtn.addEventListener("click",     () => { window.location.href = "/api/export"; });

// ── Init ──────────────────────────────────────────────────────────────────────

loadAttendance(); // populate attendance table on first page load
