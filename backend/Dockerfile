# backend/Dockerfile (use for Railway/Render/production)
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make start script executable (if it exists)
RUN if [ -f start.sh ]; then chmod +x start.sh; fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command - Explicit command for Railway
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]