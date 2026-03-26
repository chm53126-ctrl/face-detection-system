# AI-Based Face Detection Attendance System

> CVR College of Engineering — B.Tech CSE-A, III Year II Sem  
> Industry Oriented Mini Project | Batch 8

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript, WebRTC |
| Backend | Python 3.10+, Flask |
| AI Model | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV |
| Storage | CSV / Excel (openpyxl) |

---

## Project Structure

```
face-attendance/
├── app.py                    # Flask backend + YOLO inference
├── requirements.txt
├── attendance_records/       # Auto-created; daily CSV files stored here
├── templates/
│   └── index.html            # Main UI
└── static/
    ├── css/style.css
    └── js/app.js             # WebRTC capture + API calls
```

---

## Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> **YOLOv8 weight download:** On first run, `yolov8n.pt` (~6 MB) is auto-downloaded from Ultralytics.

### 3. Run the server
```bash
python app.py
```

### 4. Open the browser
```
http://localhost:5000
```

---

## How It Works

```
Student opens browser
    └─► WebRTC activates webcam
         └─► Every N seconds, canvas captures a JPEG frame
              └─► POST /api/detect  (base64 frame + student name)
                   └─► Flask decodes frame with OpenCV
                        └─► YOLOv8 detects persons (class 0)
                             ├─► Face found  → status = PRESENT
                             └─► No face     → status = ABSENT
                                  └─► Result appended to attendance_YYYY-MM-DD.csv
                                       └─► Annotated frame returned to browser
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| POST | `/api/detect` | Detect face in frame, record attendance |
| GET | `/api/attendance` | Today's summary (JSON) |
| GET | `/api/export` | Download today's CSV |
| GET | `/api/history` | List all past CSV files |

### POST /api/detect — Request body
```json
{
  "frame": "data:image/jpeg;base64,...",
  "name":  "D.Manish"
}
```

### POST /api/detect — Response
```json
{
  "status":          "PRESENT",
  "face_count":      1,
  "annotated_frame": "data:image/jpeg;base64,...",
  "timestamp":       "14:32:05"
}
```

---

## CSV Format

`attendance_records/attendance_YYYY-MM-DD.csv`

```
timestamp,name,status,face_detections
2025-04-01T14:30:00,D.Manish,PRESENT,1
2025-04-01T14:30:05,Ch.Manoj kumar,PRESENT,1
```

---

## Team

- **D. Manish** — 23B81A0528  
- **Ch. Manoj Kumar** — 23B81A0529  
- **Supervisor:** Ms. Mandadi Bhagyalaxmi, Asst. Prof., Dept. of CSE
