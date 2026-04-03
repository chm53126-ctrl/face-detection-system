"""
AI-Based Face Detection Attendance System
Backend: Python + Flask
AI Model: YOLOv8 (ultralytics)
Image Processing: OpenCV
Database: CSV/Excel
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
# Structure: { name -> { session_duration, face_detected_seconds, session_start, finalized } }
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
def decode_frame(data_url: str):
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None

def detect_faces(frame: np.ndarray):
    """Run YOLOv8 inference. Returns (face_found, count, annotated_frame)."""
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
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"

def today_csv() -> str:
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.today().isoformat()}.csv")

def append_record(name: str, status: str, face_detected_secs: float, session_duration_secs: float):
    path = today_csv()
    write_header = not os.path.exists(path)
    attendance_pct = round((face_detected_secs / session_duration_secs) * 100, 1) if session_duration_secs > 0 else 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "name", "status", "face_detected_seconds",
                        "session_duration_seconds", "attendance_percentage"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name,
            status,
            round(face_detected_secs, 1),
            round(session_duration_secs, 1),
            attendance_pct,
        ])

def load_today_records() -> dict:
    path = today_csv()
    records: dict = {}
    if not os.path.exists(path):
        return records
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            records[name] = {
                "status": row["status"],
                "timestamp": row["timestamp"],
                "face_detected_seconds": float(row.get("face_detected_seconds", 0)),
                "session_duration_seconds": float(row.get("session_duration_seconds", 0)),
                "attendance_percentage": float(row.get("attendance_percentage", 0)),
            }
    return records

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session/start", methods=["POST"])
def session_start():
    """
    Start a new attendance session for a student.
    Body: { "name": "Student Name", "session_duration": 300 }  (seconds)
    """
    data = request.get_json(force=True)
    name = data.get("name", "Unknown").strip() or "Unknown"
    session_duration = int(data.get("session_duration", 300))  # seconds

    active_sessions[name] = {
        "session_duration": session_duration,
        "face_detected_seconds": 0.0,
        "session_start": datetime.now().isoformat(),
        "finalized": False,
        "last_tick_was_face": False,
    }
    logger.info(f"Session started for {name} — duration {session_duration}s")
    return jsonify({"ok": True, "name": name, "session_duration": session_duration})


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a WebRTC frame, run YOLO, update session face-time counter.
    Body JSON: {
        "frame": "<data-url>",
        "name": "Student Name",
        "tick_seconds": 2        # how many seconds this tick represents
    }
    """
    data = request.get_json(force=True)
    frame_data = data.get("frame", "")
    name = data.get("name", "Unknown").strip() or "Unknown"
    tick_seconds = float(data.get("tick_seconds", 2))

    frame = decode_frame(frame_data)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    face_found, count, annotated = detect_faces(frame)
    status = "PRESENT" if face_found else "ABSENT"

    # Update session face-time
    session = active_sessions.get(name)
    face_detected_seconds = 0.0
    attendance_pct = 0.0
    session_duration = 0

    if session and not session["finalized"]:
        if face_found:
            session["face_detected_seconds"] += tick_seconds
        face_detected_seconds = session["face_detected_seconds"]
        session_duration = session["session_duration"]
        attendance_pct = (face_detected_seconds / session_duration * 100) if session_duration > 0 else 0

    return jsonify({
        "status": status,
        "face_count": count,
        "annotated_frame": frame_to_data_url(annotated),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "face_detected_seconds": round(face_detected_seconds, 1),
        "session_duration": session_duration,
        "attendance_percentage": round(attendance_pct, 1),
    })


@app.route("/api/session/end", methods=["POST"])
def session_end():
    """
    Finalize a session and mark attendance.
    Body: { "name": "Student Name" }
    Attendance is PRESENT if face detected >= 75% of session time.
    """
    data = request.get_json(force=True)
    name = data.get("name", "Unknown").strip() or "Unknown"

    session = active_sessions.get(name)
    if not session:
        return jsonify({"error": "No active session for this student"}), 404

    if session["finalized"]:
        return jsonify({"error": "Session already finalized"}), 400

    session["finalized"] = True

    face_secs = session["face_detected_seconds"]
    total_secs = session["session_duration"]
    pct = (face_secs / total_secs * 100) if total_secs > 0 else 0
    final_status = "PRESENT" if pct >= 75.0 else "ABSENT"

    append_record(name, final_status, face_secs, total_secs)
    del active_sessions[name]

    logger.info(f"Session ended for {name} — {pct:.1f}% face time → {final_status}")
    return jsonify({
        "name": name,
        "final_status": final_status,
        "face_detected_seconds": round(face_secs, 1),
        "session_duration": total_secs,
        "attendance_percentage": round(pct, 1),
        "threshold": 75.0,
    })


@app.route("/api/attendance")
def attendance():
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
    path = today_csv()
    if not os.path.exists(path):
        return jsonify({"error": "No records for today"}), 404
    return send_file(path, as_attachment=True,
                     download_name=f"attendance_{date.today().isoformat()}.csv")


@app.route("/api/history")
def history():
    files = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True,
    )
    return jsonify({"files": files})


# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
