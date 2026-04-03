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
CONFIG_FILE = "config.json"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "presence_threshold": 0.5   # 0.5 = 50% of call time required for PRESENT
                                 # Change to 0.75 for 75%, 0.33 for 33%, etc.
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
                # Fill in any missing keys with defaults
                for k, v in DEFAULT_CONFIG.items():
                    cfg.setdefault(k, v)
                return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ── in-memory session store ───────────────────────────────────────────────────
# Structure: { name: { "face_seconds": float, "call_start": datetime } }
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

def load_today_records() -> dict:
    path = today_csv()
    records: dict = {}
    if not os.path.exists(path):
        return records
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name not in records:
                records[name] = {
                    "status": row["status"],
                    "first_seen": row["timestamp"],
                    "last_seen": row["timestamp"],
                    "face_seconds": float(row.get("face_seconds", 0)),
                    "call_duration": float(row.get("call_duration", 0)),
                    "threshold_used": float(row.get("threshold_used", 0.5)),
                }
            else:
                records[name]["last_seen"] = row["timestamp"]
    return records

def write_final_record(name: str, status: str, face_seconds: float,
                       call_duration: float, threshold: float):
    path = today_csv()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "name", "status",
                "face_seconds", "call_duration", "threshold_used"
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name, status,
            round(face_seconds, 1),
            round(call_duration, 1),
            threshold,
        ])

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a WebRTC frame, run YOLO, return detection result + annotated frame.
    Also accumulates face_seconds in the active session.

    Body JSON:
    {
        "frame": "<data-url>",
        "name": "Student Name",
        "face_seconds": 42.5   ← cumulative face time sent from frontend timer
    }
    """
    data = request.get_json(force=True)
    frame_data = data.get("frame", "")
    name = data.get("name", "Unknown").strip() or "Unknown"
    face_seconds = float(data.get("face_seconds", 0))

    frame = decode_frame(frame_data)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    face_found, count, annotated = detect_faces(frame)
    status = "PRESENT" if face_found else "ABSENT"

    # Update session store
    if name not in active_sessions:
        active_sessions[name] = {
            "face_seconds": 0,
            "call_start": datetime.now(),
        }
    # Frontend is the authoritative timer; sync it here
    active_sessions[name]["face_seconds"] = face_seconds

    return jsonify({
        "status": status,
        "face_count": count,
        "annotated_frame": frame_to_data_url(annotated),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/api/session/start", methods=["POST"])
def session_start():
    """
    Call this when the session (class/call) begins for a student.
    Body JSON: { "name": "Student Name" }
    """
    data = request.get_json(force=True)
    name = data.get("name", "Unknown").strip() or "Unknown"
    active_sessions[name] = {
        "face_seconds": 0,
        "call_start": datetime.now(),
    }
    logger.info(f"Session started for: {name}")
    return jsonify({"message": f"Session started for {name}", "name": name})


@app.route("/api/session/end", methods=["POST"])
def session_end():
    """
    Call this when the session ends (e.g. teacher ends the class).
    Calculates final attendance based on threshold.

    Body JSON:
    {
        "name": "Student Name",
        "face_seconds": 312.4,        ← total seconds face was detected
        "call_duration_seconds": 600  ← total session length in seconds
    }

    Returns: { "name", "final_status", "face_seconds", "call_duration_seconds",
                "threshold", "required_seconds", "percentage" }
    """
    data = request.get_json(force=True)
    name = data.get("name", "Unknown").strip() or "Unknown"
    face_seconds = float(data.get("face_seconds", 0))
    call_duration = float(data.get("call_duration_seconds", 0))

    cfg = load_config()
    threshold = cfg["presence_threshold"]

    required_seconds = call_duration * threshold
    final_status = "PRESENT" if face_seconds >= required_seconds else "ABSENT"
    percentage = (face_seconds / call_duration * 100) if call_duration > 0 else 0

    # Persist final record
    write_final_record(name, final_status, face_seconds, call_duration, threshold)

    # Clean up session
    active_sessions.pop(name, None)

    logger.info(
        f"Session ended — {name}: {final_status} "
        f"({face_seconds:.0f}s / {call_duration:.0f}s, "
        f"threshold={threshold*100:.0f}%)"
    )

    return jsonify({
        "name": name,
        "final_status": final_status,
        "face_seconds": round(face_seconds, 1),
        "call_duration_seconds": round(call_duration, 1),
        "threshold": threshold,
        "required_seconds": round(required_seconds, 1),
        "percentage": round(percentage, 1),
    })


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return current attendance config."""
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def update_config():
    """
    Update attendance config settings.
    Body JSON: { "presence_threshold": 0.75 }
    presence_threshold must be between 0.0 and 1.0
    """
    data = request.get_json(force=True)
    cfg = load_config()

    if "presence_threshold" in data:
        val = float(data["presence_threshold"])
        if not (0.0 < val <= 1.0):
            return jsonify({"error": "presence_threshold must be between 0.01 and 1.0"}), 400
        cfg["presence_threshold"] = val

    save_config(cfg)
    logger.info(f"Config updated: {cfg}")
    return jsonify({"message": "Config updated", "config": cfg})


@app.route("/api/attendance")
def attendance():
    records = load_today_records()
    return jsonify({
        "date": date.today().isoformat(),
        "total": len(records),
        "present": sum(1 for r in records.values() if r["status"] == "PRESENT"),
        "absent": sum(1 for r in records.values() if r["status"] == "ABSENT"),
        "records": records,
        "config": load_config(),
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
