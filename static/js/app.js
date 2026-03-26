/* ============================================================
   AI Face Attendance System – Frontend (WebRTC + Fetch API)
   ============================================================ */

// ── State ────────────────────────────────────────────────────
let stream        = null;   // MediaStream
let detectTimer   = null;   // setInterval handle
let isDetecting   = false;

// ── DOM refs ─────────────────────────────────────────────────
const video       = document.getElementById("webcam");
const overlay     = document.getElementById("overlay");
const statusBadge = document.getElementById("status-badge");
const annotatedWrapper = document.getElementById("annotated-wrapper");
const annotatedImg     = document.getElementById("annotated-img");
const btnStart    = document.getElementById("btn-start");
const btnStop     = document.getElementById("btn-stop");
const btnSnap     = document.getElementById("btn-snap");
const nameInput   = document.getElementById("student-name");
const intervalSel = document.getElementById("interval-select");
const liveStatus  = document.getElementById("live-status");
const liveFaces   = document.getElementById("live-faces");
const liveTime    = document.getElementById("live-time");
const activityLog = document.getElementById("activity-log");

// ── Clock ────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.toLocaleTimeString("en-IN", { hour12: false });
  document.getElementById("date-display").textContent =
    now.toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
}
setInterval(updateClock, 1000);
updateClock();

// ── Camera ───────────────────────────────────────────────────
btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click",  stopCamera);
btnSnap.addEventListener("click",  () => sendFrame());

async function startCamera() {
  if (!nameInput.value.trim()) {
    alert("Please enter your name before starting.");
    nameInput.focus();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();

    btnStart.disabled = true;
    btnStop.disabled  = false;
    btnSnap.disabled  = false;
    setStatus("loading", "STARTING…");
    logEntry("info", `Camera started for ${nameInput.value.trim()}`);

    // Start periodic detection
    const interval = parseInt(intervalSel.value, 10);
    detectTimer = setInterval(sendFrame, interval);
    sendFrame(); // immediate first frame
  } catch (err) {
    alert("Camera access denied or unavailable: " + err.message);
    console.error(err);
  }
}

function stopCamera() {
  clearInterval(detectTimer);
  detectTimer = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  btnStart.disabled = false;
  btnStop.disabled  = true;
  btnSnap.disabled  = true;
  setStatus("idle", "IDLE");
  annotatedWrapper.style.display = "none";
  logEntry("info", "Camera stopped.");
  loadAttendance();
}

// ── Capture & detect ─────────────────────────────────────────
async function sendFrame() {
  if (!stream || isDetecting) return;
  isDetecting = true;
  setStatus("loading", "DETECTING…");

  try {
    // Capture frame from video onto hidden canvas
    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    const resp = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: dataUrl, name: nameInput.value.trim() || "Unknown" }),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const result = await resp.json();

    // Update UI
    const present = result.status === "PRESENT";
    setStatus(present ? "present" : "absent", result.status);
    liveStatus.textContent = result.status;
    liveStatus.style.color = present ? "var(--green)" : "var(--red)";
    liveFaces.textContent  = result.face_count;
    liveTime.textContent   = result.timestamp;

    // Show annotated frame
    annotatedImg.src = result.annotated_frame;
    annotatedWrapper.style.display = "block";

    logEntry(present ? "present" : "absent",
      `${nameInput.value.trim() || "Unknown"} — ${result.status} (${result.face_count} face(s)) at ${result.timestamp}`);

    loadAttendance();
  } catch (err) {
    console.error("Detection error:", err);
    setStatus("idle", "ERROR");
    logEntry("info", `Detection error: ${err.message}`);
  } finally {
    isDetecting = false;
  }
}

// ── Attendance dashboard ──────────────────────────────────────
async function loadAttendance() {
  try {
    const resp = await fetch("/api/attendance");
    if (!resp.ok) return;
    const data = await resp.json();

    document.getElementById("total-count").textContent   = data.total;
    document.getElementById("present-count").textContent = data.present;
    document.getElementById("absent-count").textContent  = data.absent;

    const tbody = document.getElementById("table-body");
    if (Object.keys(data.records).length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-msg">No records yet — start monitoring.</td></tr>`;
      return;
    }

    tbody.innerHTML = Object.entries(data.records).map(([name, r]) => `
      <tr>
        <td>${esc(name)}</td>
        <td><span class="${r.status === 'PRESENT' ? 'badge-present' : 'badge-absent'}">${r.status}</span></td>
        <td>${esc(r.first_seen || "—")}</td>
        <td>${esc(r.last_seen  || "—")}</td>
        <td>${r.face_count}</td>
      </tr>
    `).join("");
  } catch (err) {
    console.warn("Attendance load failed:", err);
  }
}

// ── Helpers ───────────────────────────────────────────────────
function setStatus(cls, text) {
  statusBadge.className = `status-badge ${cls}`;
  statusBadge.textContent = text;
}

function logEntry(type, msg) {
  const entry = document.createElement("div");
  entry.className = "log-entry";
  const ts = new Date().toLocaleTimeString("en-IN", { hour12: false });
  const colorClass = type === "present" ? "log-present" : type === "absent" ? "log-absent" : "";
  entry.innerHTML = `<span class="log-time">[${ts}]</span> <span class="${colorClass}">${esc(msg)}</span>`;
  activityLog.prepend(entry);
  // Keep only last 50 entries
  while (activityLog.children.length > 50) activityLog.lastChild.remove();
}

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Init ──────────────────────────────────────────────────────
loadAttendance();
setInterval(loadAttendance, 10000); // auto-refresh every 10 s
logEntry("info", "System ready. Enter name and start camera.");
