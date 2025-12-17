from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, snapshot_download
import joblib
import json
from pathlib import Path
import logging
import re
import time
from datetime import datetime, timedelta, UTC
from googleapiclient.discovery import build
import os
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Popularity Prediction API",
    description="API for predicting YouTube video popularity (Low/Medium/High) and view growth (1d, 7d, 30d)",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Path Configuration
# ==========================
# Get backend directory (main.py is in webdeploy/Backend/)
BACKEND_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BACKEND_DIR / "models"

# DeBERTa model path (inside models directory)
BERT_PATH = MODELS_DIR / "deberta_popularity_v3"
METADATA_PATH = MODELS_DIR / "growth_model_metadata.json"

logger.info(f"Backend directory: {BACKEND_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"BERT model path: {BERT_PATH}")
logger.info(f"Metadata path: {METADATA_PATH}")

# ==========================
# Device Selection: CUDA → MPS → CPU
# ==========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    logger.info("Using device: CPU")

# ==========================
# 1. Download and Load BERT Classification Model from Hugging Face
# ==========================
# Download model from Hugging Face Hub
MODEL_DIR = str(MODELS_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info("Downloading/loading model from Hugging Face Hub...")
try:
    # Download the model file from Hugging Face Hub (as shown in image)
    model_path = hf_hub_download(
        repo_id="peteoyhh/is597HCD_BertaModel",
        filename="model.safetensors",
        cache_dir=MODEL_DIR
    )
    logger.info(f"Model loaded from: {model_path}")
    
    # Download the full repository to MODEL_DIR for loading
    local_model_dir = Path(MODELS_DIR) / "is597HCD_BertaModel"
    if not local_model_dir.exists():
        snapshot_download(
            repo_id="peteoyhh/is597HCD_BertaModel",
            local_dir=str(local_model_dir)
        )
    else:
        logger.info(f"Model directory already exists: {local_model_dir}")
    
    # Load model using Auto classes (as shown in image)
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(local_model_dir))
    
except Exception as e:
    logger.warning(f"Could not download from Hugging Face Hub: {e}")
    logger.info("Falling back to local model path...")
    
    # Fallback to local model if Hugging Face download fails
    if not BERT_PATH.exists():
        raise FileNotFoundError(f"BERT model not found at {BERT_PATH} and Hugging Face download failed")

    logger.info(f"Loading model from: {BERT_PATH}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(str(BERT_PATH))
    bert_model = DebertaV2ForSequenceClassification.from_pretrained(str(BERT_PATH))

bert_model = bert_model.to(device)
bert_model.eval()
logger.info("✅ BERT model loaded successfully")

# ==========================
# 2. Download and Load Growth Models from Hugging Face
# ==========================
logger.info("Downloading/loading growth models from Hugging Face Hub...")

# Initialize local growth directory
local_growth_dir = MODELS_DIR / "growth_models"
local_growth_dir.mkdir(exist_ok=True)

# List of growth model files to download from Hugging Face
growth_model_files = [
    "best_growth_model_metadata.json",
    "best_growth_model.pkl",
    "rf_1d.pkl",
    "rf_7d.pkl",
    "rf_30d.pkl",
    "growth_model_metadata.json"  # Also download the metadata file used by the code
]

# Download growth model files from Hugging Face Hub
downloaded_growth_files = {}
try:
    for filename in growth_model_files:
        try:
            file_path = hf_hub_download(
                repo_id="peteoyhh/is597HCD_BertaModel",
                filename=filename,
                cache_dir=MODEL_DIR
            )
            downloaded_growth_files[filename] = file_path
            logger.info(f"  ✅ Downloaded {filename}")
            
            # Copy to local growth directory for easier access
            dest_path = local_growth_dir / filename
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)
                logger.info(f"  ✅ Copied {filename} to {dest_path}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not download {filename}: {e}")
    
except Exception as e:
    logger.warning(f"Could not download growth models from Hugging Face Hub: {e}")

# Determine which metadata file to use (prefer files with "best_models" key)
metadata_files = [
    local_growth_dir / "growth_model_metadata.json",  # This has the correct structure
    METADATA_PATH,  # Local fallback
    local_growth_dir / "best_growth_model_metadata.json",  # May have different structure
]

metadata_file = None
metadata = None

# Try to find a metadata file with "best_models" key
for mf in metadata_files:
    if mf.exists():
        try:
            with open(mf, "r") as f:
                test_metadata = json.load(f)
            if "best_models" in test_metadata:
                metadata_file = mf
                metadata = test_metadata
                break
        except Exception as e:
            logger.warning(f"Could not read metadata file {mf}: {e}")
            continue

# If no file with "best_models" found, use the first available one
if not metadata_file:
    for mf in metadata_files:
        if mf.exists():
            metadata_file = mf
            with open(mf, "r") as f:
                metadata = json.load(f)
            break

if not metadata_file or not metadata:
    raise FileNotFoundError(f"Metadata file not found or invalid. Tried: {[str(mf) for mf in metadata_files]}")

logger.info(f"Using metadata file: {metadata_file}")

# Check if metadata has "best_models" key, if not, assume "rf" for all targets
if "best_models" not in metadata:
    logger.warning("Metadata file does not have 'best_models' key. Assuming 'rf' for all targets.")
    metadata["best_models"] = {"1d": "rf", "7d": "rf", "30d": "rf"}

logger.info("Loading growth regression models...")

best_models = {}
for target in ["1d", "7d", "30d"]:
    model_name = metadata["best_models"][target]
    
    # Try to load from downloaded files first, then fallback to local
    model_filename = f"{model_name}_{target}.pkl"
    model_paths = [
        local_growth_dir / model_filename,
        MODELS_DIR / model_filename
    ]
    
    model_path = None
    for mp in model_paths:
        if mp.exists():
            model_path = mp
            break
    
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_filename}. Tried: {[str(mp) for mp in model_paths]}")
    
    best_models[target] = joblib.load(model_path)
    logger.info(f"  ✅ Loaded {model_name.upper()} model for {target} from {model_path}")

# Get numeric features (handle different metadata formats)
numeric_features = metadata.get("numeric_features") or metadata.get("feature_cols", [
    "title_length",
    "hashtag_count",
    "duration_sec",
    "log_duration",
    "has_description",
    "category_id"
])
embedding_dim = metadata.get("embedding_dim", 768)
logger.info(f"✅ All models loaded. Embedding dim: {embedding_dim}, Features: {numeric_features}")


# ==========================
# Request Body Models
# ==========================
class PredictRequest(BaseModel):
    title: str
    hashtags: str
    category: str
    title_length: int
    hashtag_count: int
    duration_sec: float
    log_duration: float
    has_description: int
    category_id: int


class VideoData(BaseModel):
    video_id: str
    title: str
    hashtags: str
    category: str
    category_id: int
    duration_sec: float
    log_duration: float
    has_description: int
    title_length: int
    hashtag_count: int
    views: int  # Actual views from YouTube
    published_at: str  # Published date from YouTube


class BatchPredictRequest(BaseModel):
    videos: List[VideoData]


# ==========================
# Helper: Extract BERT Embedding and Predict Popularity
# ==========================
def extract_embedding_and_predict(text: str):
    """
    Extract CLS embedding and predict popularity in one pass to avoid duplicate encoding.
    Returns: (popularity_label, popularity_probs, embedding)
    """
    encoded = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        # Get outputs with hidden states
        outputs = bert_model(**encoded, output_hidden_states=True)
        
        # Extract popularity prediction
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        idx = np.argmax(probs)
        labels = ["Low", "Medium", "High"]
        popularity_label = labels[idx]
        popularity_probs = probs.tolist()
        
        # Extract CLS embedding (last layer, first token)
        embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

    return popularity_label, popularity_probs, embedding


# ==========================
# Helper: Growth Regression
# ==========================
def build_feature_vector(req: PredictRequest, bert_embedding):
    numeric = np.array([[  
        req.title_length,
        req.hashtag_count,
        req.duration_sec,
        req.log_duration,
        req.has_description,
        req.category_id,
    ]])

    return np.hstack([bert_embedding.reshape(1, -1), numeric])


# ==========================
# API Endpoint
# ==========================
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # ----- 1) Build text for BERT (matching training format from Modeling.ipynb)
        # The training format includes structured features in the text
        text = (
            f"CATEGORY_ID: {req.category_id}. "
            f"CATEGORY: {req.category}. "
            f"TITLE_LENGTH: {req.title_length} words. "
            f"HASHTAG_COUNT: {req.hashtag_count}. "
            f"DURATION_SEC: {req.duration_sec:.1f}. "
            f"LOG_DURATION: {req.log_duration:.2f}. "
            f"HAS_DESCRIPTION: {req.has_description}. "
            f"TITLE: {req.title}. "
            f"HASHTAGS: {req.hashtags}"
        )

        # ----- 2) Extract embedding and predict popularity in one pass
        popularity_label, popularity_probs, embedding = extract_embedding_and_predict(text)

        # ----- 3) Prepare features (embedding + structured features)
        X = build_feature_vector(req, embedding)

        # ----- 4) Growth prediction
        results_growth = {}
        for target in ["1d", "7d", "30d"]:
            model = best_models[target]
            pred_log = model.predict(X)[0]
            pred_value = np.expm1(pred_log)  # reverse log1p transformation
            results_growth[target] = float(max(0, pred_value))  # Ensure non-negative

        # ----- 4.5) Apply category multiplier (use single prediction multipliers)
        multiplier = SINGLE_PREDICT_MULTIPLIERS.get(req.category, 1.0)
        results_growth_multiplied = {
            "1d": results_growth["1d"] * multiplier,
            "7d": results_growth["7d"] * multiplier,
            "30d": results_growth["30d"] * multiplier,
        }

        # ----- 5) Final output
        return {
            "popularity_label": popularity_label,
            "popularity_probabilities": {
                "Low": float(popularity_probs[0]),
                "Medium": float(popularity_probs[1]),
                "High": float(popularity_probs[2]),
            },
            "growth_prediction": {
                "views_1d": float(results_growth_multiplied["1d"]),
                "views_7d": float(results_growth_multiplied["7d"]),
                "views_30d": float(results_growth_multiplied["30d"]),
            },
            "multiplier": float(multiplier)  # Include multiplier in response for reference
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "YouTube Popularity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict video popularity and growth",
            "POST /crawl-youtube": "Crawl YouTube videos (10 categories, 3 videos each)",
            "POST /batch-predict": "Batch predict popularity and growth for multiple videos",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": list(best_models.keys()),
        "embedding_dim": embedding_dim
    }


# ==========================
# YouTube Crawler Endpoint
# ==========================
def parse_duration_iso(duration_str: str) -> float:
    """Parse ISO 8601 duration format (PT39S, PT5M10S, PT1H2M3S) to seconds."""
    if not duration_str or duration_str == "":
        return 0.0
    
    duration_str = str(duration_str).upper()
    if not duration_str.startswith('PT'):
        return 0.0
    
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0.0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


# Category mapping (matching frontend)
CATEGORY_MAPPING = {
    "comedy": {"name": "Comedy", "id": 0},
    "education": {"name": "Education", "id": 1},
    "entertainment": {"name": "Entertainment", "id": 2},
    "gaming": {"name": "Gaming", "id": 3},
    "howto style": {"name": "Howto & Style", "id": 4},
    "music": {"name": "Music", "id": 5},
    "news politics": {"name": "News & Politics", "id": 6},
    "science technology": {"name": "Science & Technology", "id": 7},
    "sports": {"name": "Sports", "id": 8},
    "travel vlog": {"name": "Travel & Events", "id": 9},
}

# Category multipliers for batch prediction
CATEGORY_MULTIPLIERS = {
    "Comedy": 33,
    "Education": 67,
    "Entertainment": 15,
    "Gaming": 180,
    "Howto & Style": 29,
    "Music": 51,
    "News & Politics": 15,
    "Science & Technology": 120,
    "Sports": 14,
    "Travel & Events": 220,
}

# Category multipliers for single prediction (different from batch)
SINGLE_PREDICT_MULTIPLIERS = {
    "Comedy": 3.75,
    "Education": 7.5,
    "Entertainment": 2,
    "Gaming": 12.5,
    "Howto & Style": 3.5,
    "Music": 7.5,
    "News & Politics": 2.25,
    "Science & Technology": 17.5,
    "Sports": 2.5,
    "Travel & Events": 37.5,
}

KEYWORDS = list(CATEGORY_MAPPING.keys())


@app.post("/crawl-youtube")
def crawl_youtube(api_key: Optional[str] = Query(None, description="YouTube Data API v3 key")):
    """
    Crawl YouTube videos from 10 categories, 3 videos per category.
    Returns video data with features needed for BERT prediction.
    """
    try:
        # Get API key from environment or parameter
        youtube_api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            raise ValueError("YouTube API key is required. Set YOUTUBE_API_KEY environment variable or pass api_key parameter.")
        
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
        
        # Time range: past 1 day
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=1)
        published_after = start_time.isoformat("T").replace("+00:00", "Z")
        published_before = end_time.isoformat("T").replace("+00:00", "Z")
        
        all_videos = []
        seen_video_ids = set()  # Track unique video IDs to prevent duplicates
        VIDEOS_PER_CATEGORY = 3  # Strict limit: exactly 3 videos per category
        
        for keyword in KEYWORDS:
            try:
                logger.info(f"🔍 Fetching videos for: {keyword}")
                
                # Search for videos - limit to 3
                search_req = youtube.search().list(
                    part="id",
                    q=keyword,
                    type="video",
                    regionCode="US",
                    publishedAfter=published_after,
                    publishedBefore=published_before,
                    order="viewCount",
                    maxResults=VIDEOS_PER_CATEGORY  # Only 3 videos per category
                )
                search_resp = search_req.execute()
                
                video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
                if not video_ids:
                    logger.warning(f"⚠️ No videos found for {keyword}")
                    continue
                
                # Limit to exactly 3 video IDs (in case API returns more)
                video_ids = video_ids[:VIDEOS_PER_CATEGORY]
                
                # Get video details
                stats_req = youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(video_ids)
                )
                stats_resp = stats_req.execute()
                
                category_info = CATEGORY_MAPPING[keyword]
                videos_added_for_category = 0  # Counter for this category
                
                for item in stats_resp.get("items", []):
                    # Strict limit: stop if we already added 3 videos for this category
                    if videos_added_for_category >= VIDEOS_PER_CATEGORY:
                        break
                    
                    video_id = item["id"]
                    
                    # Skip if we've already seen this video (prevent duplicates)
                    if video_id in seen_video_ids:
                        logger.warning(f"⚠️ Duplicate video {video_id} skipped for {keyword}")
                        continue
                    
                    stats = item.get("statistics", {}) or {}
                    snippet = item.get("snippet", {})
                    content_details = item.get("contentDetails", {})
                    
                    # Extract features
                    title = snippet.get("title", "")
                    description = snippet.get("description", "")
                    tags = snippet.get("tags", [])
                    hashtags_str = " ".join(tags) if tags else ""
                    
                    duration_str = content_details.get("duration", "")
                    duration_sec = parse_duration_iso(duration_str)
                    log_duration = np.log1p(duration_sec)
                    
                    title_length = len(title.split()) if title else 0
                    hashtag_count = len(tags) if tags else 0
                    has_description = 1 if description and description.strip() else 0
                    
                    views = int(stats.get("viewCount", 0) or 0)
                    published_at = snippet.get("publishedAt", "")
                    
                    video_data = {
                        "video_id": video_id,
                        "title": title,
                        "hashtags": hashtags_str,
                        "category": category_info["name"],
                        "category_id": category_info["id"],
                        "duration_sec": float(duration_sec),
                        "log_duration": float(log_duration),
                        "has_description": has_description,
                        "title_length": title_length,
                        "hashtag_count": hashtag_count,
                        "views": views,
                        "published_at": published_at
                    }
                    
                    all_videos.append(video_data)
                    seen_video_ids.add(video_id)  # Mark as seen
                    videos_added_for_category += 1
                
                logger.info(f"✅ Added {videos_added_for_category} videos for {keyword}")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Error fetching {keyword}: {e}")
                continue
        
        logger.info(f"✅ Successfully crawled {len(all_videos)} videos")
        return {
            "success": True,
            "count": len(all_videos),
            "videos": all_videos
        }
    
    except Exception as e:
        logger.error(f"Error during YouTube crawl: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "videos": []
        }


# ==========================
# Batch Prediction Endpoint
# ==========================
@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    """
    Batch predict popularity and view growth for multiple videos.
    Returns predictions along with actual views for comparison.
    """
    try:
        results = []
        
        for video in req.videos:
            try:
                # Build text for BERT (matching training format)
                text = (
                    f"CATEGORY_ID: {video.category_id}. "
                    f"CATEGORY: {video.category}. "
                    f"TITLE_LENGTH: {video.title_length} words. "
                    f"HASHTAG_COUNT: {video.hashtag_count}. "
                    f"DURATION_SEC: {video.duration_sec:.1f}. "
                    f"LOG_DURATION: {video.log_duration:.2f}. "
                    f"HAS_DESCRIPTION: {video.has_description}. "
                    f"TITLE: {video.title}. "
                    f"HASHTAGS: {video.hashtags}"
                )
                
                # Extract embedding and predict popularity
                popularity_label, popularity_probs, embedding = extract_embedding_and_predict(text)
                
                # Prepare features for growth prediction
                numeric = np.array([[
                    video.title_length,
                    video.hashtag_count,
                    video.duration_sec,
                    video.log_duration,
                    video.has_description,
                    video.category_id,
                ]])
                
                X = np.hstack([embedding.reshape(1, -1), numeric])
                
                # Growth prediction
                growth_pred = {}
                for target in ["1d", "7d", "30d"]:
                    model = best_models[target]
                    pred_log = model.predict(X)[0]
                    pred_value = np.expm1(pred_log)
                    growth_pred[f"views_{target}"] = float(max(0, pred_value))
                
                # Apply category multiplier to predicted views
                multiplier = CATEGORY_MULTIPLIERS.get(video.category, 1.0)
                growth_pred_multiplied = {
                    "views_1d": growth_pred["views_1d"] * multiplier,
                    "views_7d": growth_pred["views_7d"] * multiplier,
                    "views_30d": growth_pred["views_30d"] * multiplier,
                }
                
                result = {
                    "video_id": video.video_id,
                    "title": video.title,
                    "category": video.category,
                    "published_at": video.published_at,
                    "actual_views": video.views,
                    "predicted_popularity": popularity_label,
                    "popularity_probabilities": {
                        "Low": float(popularity_probs[0]),
                        "Medium": float(popularity_probs[1]),
                        "High": float(popularity_probs[2]),
                    },
                    "predicted_views": {
                        "views_1d": float(growth_pred_multiplied["views_1d"]),
                        "views_7d": float(growth_pred_multiplied["views_7d"]),
                        "views_30d": float(growth_pred_multiplied["views_30d"]),
                    },
                    "multiplier": float(multiplier)  # Include multiplier in response for reference
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting for video {video.video_id}: {e}")
                results.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}", exc_info=True)
        raise