# YouTube Popularity Predictor - Frontend

React frontend for the YouTube Popularity Prediction application.

## Features

- 📝 Interactive form to input video details
- 🔮 Real-time popularity prediction (Low/Medium/High)
- 📊 View growth forecasts for 1d, 7d, and 30d
- 🎨 Modern, responsive UI
- ⚡ Auto-computation of derived fields (title length, hashtag count, log duration)

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend API running on `http://localhost:8000` (or configure via `REACT_APP_API_URL`)

## Installation

```bash
cd webdeploy/Frontend
npm install
```

## Running the Application

```bash
npm start
```

The app will open at `http://localhost:3000`

## Configuration

### Environment Variables

Create a `.env` file in the `Frontend` directory to configure the application:

```bash
# Disable host check for development (fixes webpack-dev-server allowedHosts error)
DANGEROUSLY_DISABLE_HOST_CHECK=true

# Optional: Set backend API URL (if not using proxy)
# REACT_APP_API_URL=http://localhost:8000

# Optional: Set specific host
# HOST=localhost
```

**Note**: If you encounter the `allowedHosts` error, make sure to set `DANGEROUSLY_DISABLE_HOST_CHECK=true` in your `.env` file, or it's already configured in the start script.

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Project Structure

```
Frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.jsx       # Main application component
│   ├── App.css       # Application styles
│   ├── index.js      # Entry point
│   └── index.css     # Global styles
├── package.json
└── README.md
```

## API Integration

The frontend calls the following backend endpoints:

- `POST /predict` - Submit video data and get predictions
- `GET /health` - Health check (not used in UI, but available)

## Form Fields

- **Title** (required): Video title
- **Hashtags**: Space-separated hashtags
- **Category** (required): Video category dropdown
- **Duration** (required): Duration in seconds
- **Has Description**: Checkbox for description presence

Auto-computed fields (not user-input):
- `title_length`: Word count of title
- `hashtag_count`: Number of hashtags
- `log_duration`: log1p(duration_sec)
- `category_id`: Numeric ID mapped from category

