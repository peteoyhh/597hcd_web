#!/bin/bash

echo "🚀 Starting YouTube Popularity Prediction Frontend..."
echo ""

cd webdeploy/Frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo "✅ Starting frontend on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm start
