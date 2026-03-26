FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN mkdir -p attendance_records

EXPOSE 8080
ENV PORT=8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
