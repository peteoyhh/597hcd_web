# YouTube Popularity Predictor - Full Stack Application

Complete web application for predicting YouTube video popularity and view growth.

## Project Structure

```
webdeploy/
├── Backend/          # FastAPI backend server
│   ├── backend.py    # Main API server
│   └── requirments.txt
└── Frontend/         # React frontend application
    ├── src/
    ├── public/
    └── package.json
```

## Quick Start

### 1. Backend Setup

```bash
cd webdeploy/Backend

# Install dependencies
pip install -r requirments.txt

# Run the server
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
cd webdeploy/Frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will open at `http://localhost:3000`

## Features

### Backend API
- **POST /predict** - Predict video popularity and view growth
- **GET /health** - Health check endpoint

### Frontend
- Interactive prediction form
- Real-time field computation (title length, hashtag count, etc.)
- Beautiful result visualization
- Responsive design

## API Usage Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing Video Title",
    "hashtags": "#youtube #vlog #fun",
    "category": "Entertainment",
    "title_length": 3,
    "hashtag_count": 3,
    "duration_sec": 300,
    "log_duration": 5.704,
    "has_description": 1,
    "category_id": 2
  }'
```

## Requirements

### Backend
- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- See `Backend/requirments.txt` for full list

### Frontend
- Node.js 14+
- npm or yarn

## Development

### Backend Development
- The backend uses automatic reloading with `--reload` flag
- Logs are printed to console
- Check `/docs` endpoint for interactive API documentation

### Frontend Development
- Hot reloading enabled by default
- API proxy configured in `package.json` (points to `http://localhost:8000`)
- Set `REACT_APP_API_URL` environment variable to change API URL

## Production Deployment

### Backend
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
npm run build
# Serve the build/ directory with a static file server
```

## Notes

- Ensure model files are in the correct paths:
  - `deberta_popularity_v3/` in project root
  - `models/growth_model_metadata.json` in project root
  - Model files (`rf_1d.pkl`, etc.) in `models/` directory
- The backend automatically detects available device (CUDA/MPS/CPU)
- Frontend automatically computes derived fields (title_length, hashtag_count, log_duration)

