# Troubleshooting Guide

## Backend 404 Error

If you're getting 404 errors from the backend, follow these steps:

### 1. Check if Backend is Running

```bash
# Check if port 8000 is in use
lsof -i :8000

# Or test the connection
curl http://localhost:8000/health
```

### 2. Start the Backend Server

```bash
cd webdeploy/Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or use the provided script:
```bash
cd webdeploy/Backend
./start.sh
```

### 3. Verify Backend is Working

Open in browser:
- API Root: http://localhost:8000/
- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs

### 4. Check Frontend Configuration

Make sure `webdeploy/Frontend/src/App.jsx` has:
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
```

### 5. Check CORS Settings

Backend CORS is configured for:
- http://localhost:3000
- http://127.0.0.1:3000

If frontend runs on a different port, update `webdeploy/Backend/main.py`:
```python
allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "YOUR_PORT"],
```

### 6. Test Endpoints

Run the test script:
```bash
cd webdeploy/Backend
python3 test_endpoints.py
```

### Common Issues

1. **Port already in use**: Kill the process using port 8000
2. **Model loading fails**: Check that models are downloaded from Hugging Face
3. **CORS errors**: Check browser console for CORS-related errors
4. **Network errors**: Verify backend is accessible from frontend

