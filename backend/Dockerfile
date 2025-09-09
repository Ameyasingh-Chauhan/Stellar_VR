FROM python:3.12-slim

# Install system dependencies with minimal footprint
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Remove unnecessary documentation and locale files
    find /usr/share/doc -type f -delete 2>/dev/null || true && \
    find /usr/share/man -type f -delete 2>/dev/null || true && \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* /usr/share/locale/* 2>/dev/null || true

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Aggressive cleanup to minimize image size
    find /usr/local/lib/python3.12 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache && \
    # Remove Python test files
    find /usr/local/lib/python3.12 -name "*test*" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# Make start script executable if it exists
RUN if [ -f start.sh ]; then chmod +x start.sh; fi

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]