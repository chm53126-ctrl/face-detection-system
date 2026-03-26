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

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

ATTENDANCE_DIR = "attendance_records"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ── YOLO model ────────────────────────────────────────────────────────────────
# Uses YOLOv8n — auto-downloaded on first run (~6 MB)
model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading YOLOv8n model …")
        model = YOLO("yolov8n.pt")   # nano variant – fast, accurate enough
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
    Run YOLOv8 inference.  Returns (face_found, count, annotated_frame).
    Class 0 = 'person'; we detect persons as a proxy for face presence.
    For dedicated face detection use a face-tuned YOLO weight file.
    """
    mdl = get_model()
    results = mdl(frame, verbose=False, conf=0.45, classes=[0])  # class 0 = person
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
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(
                annotated,
                f"Person {conf:.0%}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2,
            )

    # Status banner
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
            if name not in records:
                records[name] = {
                    "status": row["status"],
                    "first_seen": row["timestamp"],
                    "last_seen": row["timestamp"],
                    "face_count": int(row.get("face_detections", 1)),
                }
            else:
                records[name]["last_seen"] = row["timestamp"]
                records[name]["face_count"] += int(row.get("face_detections", 1))
                if row["status"] == "PRESENT":
                    records[name]["status"] = "PRESENT"
    return records


def append_record(name: str, status: str, face_count: int):
    path = today_csv()
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "name", "status", "face_detections"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, status, face_count])


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a WebRTC frame (base64 data-URL), run YOLO, return result + annotated frame.
    Body JSON: { "frame": "<data-url>", "name": "Student Name" }
    """
    data = request.get_json(force=True)
    frame_data = data.get("frame", "")
    name = data.get("name", "Unknown").strip() or "Unknown"

    frame = decode_frame(frame_data)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    face_found, count, annotated = detect_faces(frame)
    status = "PRESENT" if face_found else "ABSENT"

    # Persist to CSV every detection tick
    append_record(name, status, count)

    return jsonify({
        "status": status,
        "face_count": count,
        "annotated_frame": frame_to_data_url(annotated),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/api/attendance")
def attendance():
    """Return today's attendance summary as JSON."""
    records = load_today_records()
    return jsonify({
        "date": date.today().isoformat(),
        "total": len(records),
        "present": sum(1 for r in records.values() if r["status"] == "PRESENT"),
        "absent":  sum(1 for r in records.values() if r["status"] == "ABSENT"),
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
