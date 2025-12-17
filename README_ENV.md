# Environment Setup

This project uses environment variables to store sensitive information like API keys.

## Setup Instructions

1. **Create a `.env` file** in the project root directory:
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys** to the `.env` file:
   ```
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

3. **Install required packages** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

4. **For Jupyter Notebooks**, the `.env` file will be automatically loaded when you run the notebook cells.

## Getting Your YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Go to "Credentials" and create an API key
5. Copy the API key and paste it into your `.env` file

## Security Note

- The `.env` file is already included in `.gitignore` and will NOT be committed to version control
- Never share your `.env` file or commit it to Git
- The `.env.example` file shows the format without exposing actual keys
