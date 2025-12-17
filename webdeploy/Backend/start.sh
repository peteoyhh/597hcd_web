#!/bin/bash

# Start the FastAPI backend server
cd "$(dirname "$0")"

echo "Starting YouTube Popularity Prediction API..."
echo "Backend will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000

