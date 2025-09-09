#!/usr/bin/env bash

# Start script for STELLAR VR180 Backend

# Set default values if not already set
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

echo "🚀 Starting STELLAR VR180 Backend..."
echo "📍 Host: $HOST"
echo "📍 Port: $PORT"

# Start the application
exec uvicorn app.main:app --host $HOST --port $PORT