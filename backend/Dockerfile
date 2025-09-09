FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefer-binary -r requirements.txt && rm -rf /root/.cache

# Copy application code
COPY . .

# Make start script executable if it exists
RUN if [ -f start.sh ]; then chmod +x start.sh; fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]