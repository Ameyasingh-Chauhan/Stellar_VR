# backend/Dockerfile (use for Railway/Render/production)
FROM python:3.12-slim

# system deps
RUN apt-get update && apt-get install -y ffmpeg build-essential git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# NOTE: install torch separately if you want GPU in container (use appropriate wheel)
# For CPU-only, you can install torch here after verifying the wheel; otherwise install manually.

COPY . /app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]