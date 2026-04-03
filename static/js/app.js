/**
 * FaceAttend AI — Frontend Logic
 * 
 * Flow:
 * 1. Student enters name and selects session duration (2/5/10/30 min)
 * 2. "Start Session" → camera opens, /api/session/start called, countdown begins
 * 3. Every TICK_MS: capture frame → POST /api/detect
 *    - If face detected → timer continues, face-time accumulates
 *    - If no face     → timer PAUSES (ring turns amber)
 * 4. Session ends (timer reaches 0 OR user clicks "End Early")
 *    → POST /api/session/end → server evaluates 75% rule → returns final status
 * 5. Result card shown; attendance table refreshed
 */

// ── Constants ──────────────────────────────────────────────────────────────
const TICK_MS       = 2000;   // detection interval in ms
const TICK_SECONDS  = TICK_MS / 1000;
const RING_CIRCUM   = 326.7;  // 2π × 52 (SVG circle r=52)

// ── DOM refs ───────────────────────────────────────────────────────────────
const studentNameEl  = document.getElementById("studentName");
const sessionBtns    = document.querySelectorAll(".session-btn");
const startBtn       = document.getElementById("startBtn");
const endBtn         = document.getElementById("endBtn");
const newSessionBtn  = document.getElementById("newSessionBtn");
const refreshBtn     = document.getElementById("refreshBtn");
const exportBtn      = document.getElementById("exportBtn");

const setupCard      = document.getElementById("setupCard");
const sessionCard    = document.getElementById("sessionCard");
const resultCard     = document.getElementById("resultCard");

const timerDisplay   = document.getElementById("timerDisplay");
const ringProgress   = document.getElementById("ringProgress");
const presenceFill   = document.getElementById("presenceFill");
const presencePct    = document.getElementById("presencePct");
const statusPill     = document.getElementById("statusPill");
const pillDot        = document.getElementById("pillDot");
const pillText       = document.getElementById("pillText");
const recDot         = document.getElementById("recDot");
const cameraIdle     = document.getElementById("cameraIdle");

const resultBadge    = document.getElementById("resultBadge");
const resFaceTime    = document.getElementById("resFaceTime");
const resSessionTime = document.getElementById("resSessionTime");
const resPct         = document.getElementById("resPct");
const attBody        = document.getElementById("attBody");
const attSummary     = document.getElementById("attSummary");

const video          = document.getElementById("video");
const canvas         = document.getElementById("overlay");
const ctx            = canvas.getContext("2d");

// ── State ──────────────────────────────────────────────────────────────────
let selectedDuration  = 0;       // seconds
let sessionActive     = false;
let sessionName       = "";

let remainingSeconds  = 0;       // counts DOWN (wall-clock view)
let faceSeconds       = 0;       // total seconds face was detected
let sessionDuration   = 0;       // total session seconds

let countdownTimer    = null;    // setInterval for 1-second countdown
let detectionTimer    = null;    // setInterval for YOLO ticks
let faceCurrentlyOn   = false;   // is face detected right now?

let stream            = null;    // MediaStream

// ── Session button selection ───────────────────────────────────────────────
sessionBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    sessionBtns.forEach(b => b.classList.remove("selected"));
    btn.classList.add("selected");
    selectedDuration = parseInt(btn.dataset.seconds, 10);
    validateStart();
  });
});

studentNameEl.addEventListener("input", validateStart);

function validateStart() {
  startBtn.disabled = !(studentNameEl.value.trim() && selectedDuration > 0);
}

// ── Start session ──────────────────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
  sessionName     = studentNameEl.value.trim();
  sessionDuration = selectedDuration;
  remainingSeconds = sessionDuration;
  faceSeconds     = 0;

  // Tell server about this session
  try {
    await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: sessionName, session_duration: sessionDuration }),
    });
  } catch (e) {
    alert("Could not contact server. Is app.py running?");
    return;
  }

  // Open camera
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    cameraIdle.style.display = "none";
    recDot.classList.add("active");
  } catch (e) {
    alert("Camera permission denied. Please allow camera access.");
    return;
  }

  // Swap cards
  setupCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  sessionCard.classList.remove("hidden");

  sessionActive = true;
  updateTimerUI();
  updatePresenceBar(0);

  // 1-second countdown tick (only ticks when face is on)
  countdownTimer = setInterval(() => {
    if (!sessionActive) return;
    if (!faceCurrentlyOn) return;      // PAUSE countdown if no face
    remainingSeconds--;
    updateTimerUI();
    if (remainingSeconds <= 0) endSession();
  }, 1000);

  // Detection tick every TICK_MS
  detectionTimer = setInterval(runDetection, TICK_MS);
  runDetection(); // immediate first tick
});

// ── Detection tick ─────────────────────────────────────────────────────────
async function runDetection() {
  if (!sessionActive) return;

  // Capture frame from webcam
  const offscreen = document.createElement("canvas");
  offscreen.width  = video.videoWidth  || 640;
  offscreen.height = video.videoHeight || 480;
  offscreen.getContext("2d").drawImage(video, 0, 0);
  const frameData = offscreen.toDataURL("image/jpeg", 0.8);

  let data;
  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame: frameData,
        name: sessionName,
        tick_seconds: TICK_SECONDS,
      }),
    });
    data = await res.json();
  } catch (e) {
    console.error("Detection failed:", e);
    return;
  }

  if (!sessionActive) return;

  // Update face state
  faceCurrentlyOn = (data.status === "PRESENT");
  faceSeconds = data.face_detected_seconds ?? faceSeconds;

  // Draw annotated frame on overlay canvas
  if (data.annotated_frame) {
    const img = new Image();
    img.onload = () => {
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = data.annotated_frame;
  }

  // Update presence pill
  const pct = sessionDuration > 0 ? (faceSeconds / sessionDuration) * 100 : 0;
  updatePresenceBar(pct);
  updatePill(faceCurrentlyOn);

  // Update ring color: amber when face absent (timer paused)
  if (!faceCurrentlyOn) {
    ringProgress.classList.add("paused");
  } else {
    ringProgress.classList.remove("paused");
  }
}

// ── End session ────────────────────────────────────────────────────────────
endBtn.addEventListener("click", () => endSession());

async function endSession() {
  if (!sessionActive) return;
  sessionActive = false;

  clearInterval(countdownTimer);
  clearInterval(detectionTimer);

  // Stop camera
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  recDot.classList.remove("active");
  cameraIdle.style.display = "flex";
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Finalize on server
  let result;
  try {
    const res = await fetch("/api/session/end", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: sessionName }),
    });
    result = await res.json();
  } catch (e) {
    alert("Could not finalize session on server.");
    showSetup();
    return;
  }

  // Show result card
  sessionCard.classList.add("hidden");
  resultCard.classList.remove("hidden");

  const isPresent = result.final_status === "PRESENT";
  resultBadge.textContent   = isPresent ? "✓ PRESENT" : "✗ ABSENT";
  resultBadge.className     = "result-badge " + (isPresent ? "present" : "absent");
  resFaceTime.textContent   = fmtSecs(result.face_detected_seconds);
  resSessionTime.textContent = fmtSecs(result.session_duration);
  resPct.textContent        = result.attendance_percentage.toFixed(1) + "%";

  loadAttendance();
}

// ── New session button ─────────────────────────────────────────────────────
newSessionBtn.addEventListener("click", showSetup);
function showSetup() {
  resultCard.classList.add("hidden");
  sessionCard.classList.add("hidden");
  setupCard.classList.remove("hidden");
  // reset UI
  sessionBtns.forEach(b => b.classList.remove("selected"));
  selectedDuration = 0;
  startBtn.disabled = true;
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function updateTimerUI() {
  const pctLeft = remainingSeconds / sessionDuration;
  const offset  = RING_CIRCUM * (1 - pctLeft);
  ringProgress.style.strokeDashoffset = offset;
  timerDisplay.textContent = fmtSecs(remainingSeconds);
}

function updatePresenceBar(pct) {
  const clamped = Math.min(pct, 100);
  presenceFill.style.width = clamped + "%";
  presencePct.textContent  = clamped.toFixed(1) + "%";
  if (clamped >= 75) {
    presenceFill.classList.remove("danger");
  } else {
    presenceFill.classList.add("danger");
  }
}

function updatePill(faceOn) {
  statusPill.className = "status-pill " + (faceOn ? "present" : "absent");
  pillText.textContent = faceOn ? "Face Detected — Timer Running" : "No Face — Timer Paused";
}

function fmtSecs(s) {
  s = Math.round(s);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
}

// ── Attendance table ────────────────────────────────────────────────────────
refreshBtn.addEventListener("click", loadAttendance);
exportBtn.addEventListener("click", () => { window.location.href = "/api/export"; });

async function loadAttendance() {
  try {
    const res  = await fetch("/api/attendance");
    const data = await res.json();
    renderTable(data);
  } catch (e) {
    console.error("Failed to load attendance:", e);
  }
}

function renderTable(data) {
  const records = data.records || {};
  const names   = Object.keys(records);

  if (!names.length) {
    attBody.innerHTML = `<tr class="empty-row"><td colspan="6">No records yet</td></tr>`;
    attSummary.innerHTML = "";
    return;
  }

  attBody.innerHTML = names.map(name => {
    const r = records[name];
    const badge = `<span class="status-badge ${r.status}">${r.status}</span>`;
    const faceSecs = r.face_detected_seconds != null ? fmtSecs(r.face_detected_seconds) : "—";
    const sessSecs = r.session_duration_seconds != null ? fmtSecs(r.session_duration_seconds) : "—";
    const pct = r.attendance_percentage != null ? r.attendance_percentage.toFixed(1) + "%" : "—";
    const time = r.timestamp ? r.timestamp.split("T")[1]?.substring(0,8) : "—";
    return `<tr>
      <td>${name}</td>
      <td>${badge}</td>
      <td>${faceSecs}</td>
      <td>${sessSecs}</td>
      <td>${pct}</td>
      <td>${time}</td>
    </tr>`;
  }).join("");

  attSummary.innerHTML = `
    <span>Total: <strong>${data.total}</strong></span>
    <span style="color:var(--accent)">Present: <strong>${data.present}</strong></span>
    <span style="color:var(--danger)">Absent: <strong>${data.absent}</strong></span>
  `;
}

// ── Init ────────────────────────────────────────────────────────────────────
loadAttendance();
