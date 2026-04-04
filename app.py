"""
AI-Based Face Detection Attendance System
Backend: Python + Flask
AI Model: YOLOv8 (ultralytics)
Image Processing: OpenCV
Database: CSV/Excel

UPDATED: Session-based attendance with 75% presence threshold.
- Session timer runs for the full configured duration.
- Frames are captured and checked throughout the entire session.
- At session end, if face was detected in >= 75% of frames → PRESENT, else ABSENT.
"""

import os
import base64
import csv
import json
import logging
from datetime import datetime, date
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

ATTENDANCE_DIR = "attendance_records"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ── In-memory session store ───────────────────────────────────────────────────
# Structure: { session_id: { "name": str, "total_frames": int, "face_frames": int, "start_time": datetime } }
active_sessions: dict = {}

# ── YOLO model ────────────────────────────────────────────────────────────────
model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading YOLOv8n model …")
        model = YOLO("yolov8n.pt")
        logger.info("Model ready.")
    return model

# ── helpers ───────────────────────────────────────────────────────────────────

def decode_frame(data_url: str) -> np.ndarray | None:
    """Decode a base64 data-URL coming from WebRTC canvas into an OpenCV image."""
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


def detect_faces(frame: np.ndarray) -> tuple[bool, int, np.ndarray]:
    """
    Run YOLOv8 inference. Returns (face_found, count, annotated_frame).
    Class 0 = 'person'; we detect persons as a proxy for face presence.
    """
    mdl = get_model()
    results = mdl(frame, verbose=False, conf=0.45, classes=[0])
    count = 0
    annotated = frame.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(
                annotated,
                f"Person {conf:.0%}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2,
            )

    status_text = f"{'PRESENT' if count > 0 else 'ABSENT'} — {count} detected"
    color = (0, 200, 80) if count > 0 else (0, 60, 220)
    cv2.rectangle(annotated, (0, 0), (frame.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(annotated, status_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return count > 0, count, annotated


def frame_to_data_url(frame: np.ndarray) -> str:
    """Encode annotated OpenCV frame back to a base64 JPEG data-URL."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"


def today_csv() -> str:
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.today().isoformat()}.csv")


def load_today_records() -> dict:
    """Return {name: {status, timestamps, face_count}} for today."""
    path = today_csv()
    records: dict = {}
    if not os.path.exists(path):
        return records
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            records[name] = {
                "status": row["status"],
                "session_start": row.get("session_start", ""),
                "session_end": row.get("session_end", ""),
                "face_frames": int(row.get("face_frames", 0)),
                "total_frames": int(row.get("total_frames", 0)),
                "presence_pct": float(row.get("presence_pct", 0)),
            }
    return records


def write_final_record(name: str, status: str, session_start: str, session_end: str,
                        face_frames: int, total_frames: int, presence_pct: float):
    """Write the final attendance result after the session ends."""
    path = today_csv()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "session_start", "session_end", "name", "status",
                "face_frames", "total_frames", "presence_pct"
            ])
        w.writerow([
            session_start, session_end, name, status,
            face_frames, total_frames, f"{presence_pct:.1f}"
        ])

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start-session", methods=["POST"])
def start_session():
    """
    Called when the student submits their name and session duration.
    Initialises a session tracking entry.

    Body JSON: { "name": "Student Name", "session_id": "unique-id" }
    """
    data = request.get_json(force=True)
    name = data.get("name", "Unknown").strip() or "Unknown"
    session_id = data.get("session_id", "")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    active_sessions[session_id] = {
        "name": name,
        "total_frames": 0,
        "face_frames": 0,
        "start_time": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info(f"Session started: {session_id} for {name}")
    return jsonify({"message": "Session started", "session_id": session_id, "name": name})


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a WebRTC frame during an active session.
    Records whether a face was detected in this frame.
    The session timer on the frontend runs independently — frames are sent
    throughout the ENTIRE session regardless of detection result.

    Body JSON: { "frame": "<data-url>", "session_id": "unique-id" }
    """
    data = request.get_json(force=True)
    frame_data = data.get("frame", "")
    session_id = data.get("session_id", "")

    if session_id not in active_sessions:
        return jsonify({"error": "No active session. Call /api/start-session first."}), 400

    frame = decode_frame(frame_data)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    face_found, count, annotated = detect_faces(frame)

    # Update running totals for this session
    session = active_sessions[session_id]
    session["total_frames"] += 1
    if face_found:
        session["face_frames"] += 1

    total = session["total_frames"]
    face = session["face_frames"]
    current_pct = (face / total * 100) if total > 0 else 0

    return jsonify({
        "face_found": face_found,
        "face_count": count,
        "annotated_frame": frame_to_data_url(annotated),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        # Live stats so the UI can show running percentage
        "session_stats": {
            "total_frames": total,
            "face_frames": face,
            "presence_pct": round(current_pct, 1),
        }
    })


@app.route("/api/end-session", methods=["POST"])
def end_session():
    """
    Called when the session timer expires on the frontend.
    Computes final attendance status using the 75% threshold and writes the CSV.

    Body JSON: { "session_id": "unique-id" }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")

    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404

    session = active_sessions.pop(session_id)  # remove from active

    name = session["name"]
    total_frames = session["total_frames"]
    face_frames = session["face_frames"]
    start_time = session["start_time"]
    end_time = datetime.now().isoformat(timespec="seconds")

    # ── 75% rule ──────────────────────────────────────────────────────────────
    if total_frames == 0:
        presence_pct = 0.0
    else:
        presence_pct = (face_frames / total_frames) * 100

    status = "PRESENT" if presence_pct >= 75.0 else "ABSENT"
    # ──────────────────────────────────────────────────────────────────────────

    write_final_record(
        name=name,
        status=status,
        session_start=start_time,
        session_end=end_time,
        face_frames=face_frames,
        total_frames=total_frames,
        presence_pct=presence_pct,
    )

    logger.info(
        f"Session ended: {name} | {face_frames}/{total_frames} frames "
        f"({presence_pct:.1f}%) → {status}"
    )

    return jsonify({
        "name": name,
        "status": status,
        "face_frames": face_frames,
        "total_frames": total_frames,
        "presence_pct": round(presence_pct, 1),
        "session_start": start_time,
        "session_end": end_time,
        "message": f"{name} marked as {status} ({presence_pct:.1f}% face presence)"
    })


@app.route("/api/attendance")
def attendance():
    """Return today's attendance summary as JSON."""
    records = load_today_records()
    return jsonify({
        "date": date.today().isoformat(),
        "total": len(records),
        "present": sum(1 for r in records.values() if r["status"] == "PRESENT"),
        "absent": sum(1 for r in records.values() if r["status"] == "ABSENT"),
        "records": records,
    })


@app.route("/api/export")
def export():
    """Download today's CSV."""
    path = today_csv()
    if not os.path.exists(path):
        return jsonify({"error": "No records for today"}), 404
    return send_file(path, as_attachment=True,
                     download_name=f"attendance_{date.today().isoformat()}.csv")


@app.route("/api/history")
def history():
    """List all past attendance CSV files."""
    files = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True,
    )
    return jsonify({"files": files})


# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
