#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

df = pd.read_csv("./data/CLEANED_MERGED.csv")
print(df.columns)


# In[9]:


import numpy as np

# Ensure views, likes, comments exist, fill with 0 if missing
for col in ["views", "likes", "comments"]:
    if col not in df.columns:
        df[col] = 0

# Avoid division by zero
df["views_safe"] = df["views"].replace(0, np.nan)

df["like_rate"] = df["likes"] / df["views_safe"]
df["comment_rate"] = df["comments"] / df["views_safe"]

df["like_rate"] = df["like_rate"].fillna(0)
df["comment_rate"] = df["comment_rate"].fillna(0)


# In[10]:


df["popularity_raw"] = (
    df["views"] +
    5 * df["likes"] +
    10 * df["comments"] 
)


# In[11]:


min_raw = df["popularity_raw"].min()
max_raw = df["popularity_raw"].max()

df["popularity_score"] = (df["popularity_raw"] - min_raw) / (max_raw - min_raw)


# In[13]:


import re
import numpy as np
import pandas as pd


# --- 1. Parse duration PTxxHxxMxxS → seconds ---
def parse_duration(duration_str):
    if not isinstance(duration_str, str):
        return np.nan
    
    pattern = r"PT((?P<h>\d+)H)?((?P<m>\d+)M)?((?P<s>\d+)S)?"
    m = re.match(pattern, duration_str)
    if not m:
        return np.nan
    
    h = int(m.group("h")) if m.group("h") else 0
    m_ = int(m.group("m")) if m.group("m") else 0
    s = int(m.group("s")) if m.group("s") else 0
    return h * 3600 + m_ * 60 + s

# --- 2. Feature Engineering ---
df["duration_sec"] = df["duration"].apply(parse_duration)

df["title_length"] = df["title"].fillna("").apply(lambda x: len(x.split()))

df["hashtag_count"] = df["hashtags"].fillna("").apply(
    lambda x: len([h for h in str(x).split() if h.startswith("#")])
)

df["log_duration"] = np.log1p(df["duration_sec"])

df["has_description"] = df["description"].fillna("").apply(lambda x: 1 if len(x.strip()) > 0 else 0)

df[["duration", "duration_sec", "log_duration", "title_length", "hashtag_count", "has_description"]].head()




df["popularity_label"] = pd.qcut(
    df["popularity_score"],
    q=3,
    labels=["Low", "Medium", "High"]
)










df.to_csv("./data/CLEANED_MERGED_with_popularity.csv", index=False)



# In[42]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

source_df = pd.read_csv("./data/CLEANED_MERGED_with_popularity.csv")

source_df = source_df.dropna(subset=["title", "popularity_label"])
source_df = source_df[source_df["title"].str.strip() != ""]

print(f"Step 1: Loaded source data")
print(f"  Source rows: {len(source_df)}")
print(f"  Unique video_ids: {source_df['video_id'].nunique()}")

video_counts = source_df.groupby('video_id').size()
print(f"  Average rows per video: {video_counts.mean():.1f}")
print(f"  Videos with multiple rows: {(video_counts > 1).sum()}")

train_video_ids = []
test_video_ids = []

print(f"\nStep 2: Split video_ids by category (1:9 ratio)")
np.random.seed(42)

for category in source_df["category"].dropna().unique():
    cat_rows = source_df[source_df["category"] == category]
    cat_videos = cat_rows["video_id"].unique().tolist()
    
    if len(cat_videos) == 0:
        continue
    
    if len(cat_videos) == 1:
        train_video_ids.append(cat_videos[0])
        continue
    
    cat_train_vids, cat_test_vids = train_test_split(
        cat_videos,
        test_size=0.10,
        random_state=42,
        shuffle=True
    )
    
    train_video_ids.extend(cat_train_vids)
    test_video_ids.extend(cat_test_vids)

print(f"  Assigned {len(train_video_ids)} videos to train")
print(f"  Assigned {len(test_video_ids)} videos to test")

print(f"\nStep 3: Extract all rows from source_df based on video_id")
train_df = source_df[source_df["video_id"].isin(train_video_ids)].copy()
test_df = source_df[source_df["video_id"].isin(test_video_ids)].copy()

print(f"  Train rows before dedup: {len(train_df)}")
print(f"  Test rows before dedup: {len(test_df)}")

print(f"\nStep 4: Deduplicate by crawl_date")
if 'crawl_date' in train_df.columns:
    train_df = train_df.drop_duplicates(subset=['video_id', 'crawl_date'], keep='first')

if 'crawl_date' in test_df.columns:
    test_df = test_df.drop_duplicates(subset=['video_id', 'crawl_date'], keep='first')

print(f"\n{'='*60}")
print(f"Final Split Summary")
print(f"{'='*60}")
print(f"Source: {len(source_df)} rows, {source_df['video_id'].nunique()} unique video_ids")
print(f"Train: {len(train_df)} rows, {train_df['video_id'].nunique()} unique video_ids ({len(train_df)/len(source_df)*100:.2f}% of rows)")
print(f"Test: {len(test_df)} rows, {test_df['video_id'].nunique()} unique video_ids ({len(test_df)/len(source_df)*100:.2f}% of rows)")

overlap = set(train_video_ids) & set(test_video_ids)
print(f"Overlap video_ids: {len(overlap)} (should be 0)")

print(f"\nPer-category summary:")
category_summary = []
for category in sorted(source_df["category"].dropna().unique()):
    cat_source = source_df[source_df["category"] == category]
    total_videos = cat_source["video_id"].nunique()
    total_rows = len(cat_source)
    
    train_rows = len(train_df[train_df["category"] == category])
    test_rows = len(test_df[test_df["category"] == category])
    
    train_videos = train_df[train_df["category"] == category]["video_id"].nunique()
    test_videos = test_df[test_df["category"] == category]["video_id"].nunique()
    
    video_split_pct = (test_videos / total_videos * 100) if total_videos > 0 else 0
    
    category_summary.append({
        "category": category,
        "total_videos": total_videos,
        "train_videos": train_videos,
        "test_videos": test_videos,
        "video_test_pct": f"{video_split_pct:.1f}%",
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows
    })

summary_df = pd.DataFrame(category_summary)
display_cols = ["category", "total_videos", "train_videos", "test_videos", "video_test_pct", "total_rows", "train_rows", "test_rows"]
print(summary_df[display_cols].to_string(index=False))

train_df.to_csv("./data/train_data.csv", index=False)
test_df.to_csv("./data/test_data.csv", index=False)

print(f"\n✅ Saved train_data.csv ({len(train_df)} rows)")
print(f"✅ Saved test_data.csv ({len(test_df)} rows)")

train_video_counts = train_df.groupby('video_id').size()
test_video_counts = test_df.groupby('video_id').size()

print(f"\n📊 Video Statistics:")
print(f"  Average rows per video (train): {train_video_counts.mean():.1f}")
print(f"  Average rows per video (test): {test_video_counts.mean():.1f}")
print(f"  Videos with multiple rows (train): {(train_video_counts > 1).sum()}")
print(f"  Videos with multiple rows (test): {(test_video_counts > 1).sum()}")


# In[35]:


"""
#This part is for the time -series study, Regulary Bert Classfier won't need this. 

import pandas as pd

test_df = pd.read_csv("./data/test_data.csv")
source_df = pd.read_csv("./data/CLEANED_MERGED_with_popularity.csv")

# Remove all columns with _source suffix (leftover from merge operation)
test_df = test_df[[col for col in test_df.columns if not col.endswith('_source')]]

print(f"Test data before: {len(test_df)} rows, {test_df['video_id'].nunique()} unique video_ids")
print(f"Source data: {len(source_df)} rows, {source_df['video_id'].nunique()} unique video_ids")

test_video_ids = set(test_df["video_id"].unique())
source_rows_to_add = source_df[source_df["video_id"].isin(test_video_ids)].copy()

# Only keep columns that exist in test_df to avoid adding new columns
existing_columns = test_df.columns.tolist()
source_rows_to_add = source_rows_to_add[[col for col in existing_columns if col in source_rows_to_add.columns]]

test_df = pd.concat([test_df, source_rows_to_add], ignore_index=True)

if 'crawl_date' in test_df.columns:
    test_df = test_df.drop_duplicates(subset=['video_id', 'crawl_date'], keep='first')

print(f"\nTest data after: {len(test_df)} rows, {test_df['video_id'].nunique()} unique video_ids")

test_df = test_df.sort_values(by=['video_id', 'crawl_date'])

test_df.to_csv("./data/test_data.csv", index=False)
print(f"\n✅ Updated test_data.csv ({len(test_df)} rows)")


# In[36]:


"""

#This part is for the time -series study, Regulary Bert Classfier won't need this. 

import pandas as pd

train_df = pd.read_csv("./data/train_data.csv")
source_df = pd.read_csv("./data/CLEANED_MERGED_with_popularity.csv")

# Remove all columns with _source suffix (leftover from merge operation)
train_df = train_df[[col for col in train_df.columns if not col.endswith('_source')]]

print(f"Train data before: {len(train_df)} rows, {train_df['video_id'].nunique()} unique video_ids")
print(f"Source data: {len(source_df)} rows, {source_df['video_id'].nunique()} unique video_ids")

train_video_ids = set(train_df["video_id"].unique())
source_rows_to_add = source_df[source_df["video_id"].isin(train_video_ids)].copy()

# Only keep columns that exist in train_df to avoid adding new columns
existing_columns = train_df.columns.tolist()
source_rows_to_add = source_rows_to_add[[col for col in existing_columns if col in source_rows_to_add.columns]]

train_df = pd.concat([train_df, source_rows_to_add], ignore_index=True)

if 'crawl_date' in train_df.columns:
    train_df = train_df.drop_duplicates(subset=['video_id', 'crawl_date'], keep='first')

print(f"\nTrain data after: {len(train_df)} rows, {train_df['video_id'].nunique()} unique video_ids")

train_df = train_df.sort_values(by=['video_id', 'crawl_date'])

train_df.to_csv("./data/train_data.csv", index=False)
print(f"\n✅ Updated train_data.csv ({len(train_df)} rows)")


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
import torch.nn as nn
from datasets import Dataset

# Check and set device (prefer MPS for M1, then CUDA, finally CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU only)")

train_df = pd.read_csv("./data/train_data.csv")

# (1) Enhanced text with structured features + (2) Category-aware with category_id
def build_combined_text(row):
    # Category ID factorization (category-aware modeling)
    category_id = int(row["category_id"]) if pd.notna(row["category_id"]) else -1
    category = str(row["category"]) if pd.notna(row["category"]) else "Unknown"
    
    # Structured features
    title_len = int(row["title_length"]) if pd.notna(row["title_length"]) else 0
    hashtag_count = int(row["hashtag_count"]) if pd.notna(row["hashtag_count"]) else 0
    duration_sec = float(row["duration_sec"]) if pd.notna(row["duration_sec"]) else 0.0
    log_duration = float(row["log_duration"]) if pd.notna(row["log_duration"]) else 0.0
    has_desc = 1 if pd.notna(row["description"]) and str(row["description"]).strip() != "" else 0
    
    # Text fields
    title = str(row["title"]) if pd.notna(row["title"]) else ""
    hashtags = str(row["hashtags"]) if pd.notna(row["hashtags"]) and str(row["hashtags"]).lower() != "nan" else ""
    
    # Combined text with structured features injected
    combined = (
        f"CATEGORY_ID: {category_id}. "
        f"CATEGORY: {category}. "
        f"TITLE_LENGTH: {title_len} words. "
        f"HASHTAG_COUNT: {hashtag_count}. "
        f"DURATION_SEC: {duration_sec:.1f}. "
        f"LOG_DURATION: {log_duration:.2f}. "
        f"HAS_DESCRIPTION: {has_desc}. "
        f"TITLE: {title}. "
        f"HASHTAGS: {hashtags}"
    )
    return combined

train_df["text"] = train_df.apply(build_combined_text, axis=1)

label_mapping = {"Low": 0, "Medium": 1, "High": 2}
train_df["label"] = train_df["popularity_label"].map(label_mapping)
train_df = train_df[train_df["label"].notna()]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"]
)

model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

class YouTubeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = YouTubeDataset(train_encodings, train_labels)
val_dataset = YouTubeDataset(val_encodings, val_labels)

num_labels = 3
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

id2label = {0: "Low", 1: "Medium", 2: "High"}
label2id = {"Low": 0, "Medium": 1, "High": 2}
model.config.id2label = id2label
model.config.label2id = label2id

# Move model to device
model = model.to(device)
print(f"Model moved to {device}")

# (3) Compute class weights for handling class imbalance
label_counts = train_df["label"].value_counts().sort_index()
num_classes = len(label_counts)
total_samples = len(train_df)

# Inverse frequency weighting: weight_c = N_total / (num_classes * count_c)
class_weights = torch.tensor([
    total_samples / (num_classes * label_counts.get(i, 1))
    for i in range(num_classes)
], dtype=torch.float32).to(device)

print(f"\nClass distribution:")
for i, label_name in enumerate(["Low", "Medium", "High"]):
    count = label_counts.get(i, 0)
    weight = class_weights[i].item()
    print(f"  {label_name} (label {i}): count={count}, weight={weight:.4f}")

# (3) Custom Trainer with weighted loss + (4) Optional focal loss support
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if not use_focal_loss:
            # Standard weighted cross-entropy loss
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # For focal loss, we'll compute it manually
            self.loss_fn = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Handle num_items_in_batch and other kwargs that newer Trainer versions may pass
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if not self.use_focal_loss:
            # Standard weighted cross-entropy
            loss = self.loss_fn(logits, labels)
        else:
            # Focal loss: FL = -alpha * (1-p_t)^gamma * log(p_t)
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')(logits, labels)
            pt = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
            loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

# Set training parameters based on device type
use_fp16 = torch.cuda.is_available()  # MPS doesn't fully support fp16 yet, only enable on CUDA
use_pin_memory = torch.cuda.is_available()  # MPS doesn't support pin_memory

training_args = TrainingArguments(
    output_dir="./deberta_popularity_v3",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=use_fp16,  # Only enable mixed precision training on CUDA
    dataloader_pin_memory=use_pin_memory  # Only enable pin_memory on CUDA
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Use custom WeightedTrainer instead of default Trainer
# Set use_focal_loss=True to enable focal loss (optional, default is weighted CE)
USE_FOCAL_LOSS = False  # Set to True to enable focal loss
trainer = WeightedTrainer(
    class_weights=class_weights,
    use_focal_loss=USE_FOCAL_LOSS,
    focal_alpha=0.25,  # Focal loss alpha parameter
    focal_gamma=2.0,   # Focal loss gamma parameter
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

val_results = trainer.evaluate()
print("Validation Results (on training split):", val_results)

model.save_pretrained("./deberta_popularity_v3")
tokenizer.save_pretrained("./deberta_popularity_v3")
print("Model saved to ./deberta_popularity_v3")


# In[19]:


def predict_popularity(title, hashtags, category, model, tokenizer):
    combined_text = f"CATEGORY: {category}. TITLE: {title}. HASHTAGS: {hashtags}"
    
    inputs = tokenizer(combined_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    
    # Move inputs to the device where model is located
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Move to CPU first, then convert to numpy
    probs = probabilities[0].cpu().numpy()
    label_idx = np.argmax(probs)
    
    label_mapping = {0: "Low", 1: "Medium", 2: "High"}
    predicted_label = label_mapping[label_idx]
    
    prob_dict = {
        "Low": float(probs[0]),
        "Medium": float(probs[1]),
        "High": float(probs[2])
    }
    
    return predicted_label, prob_dict


# In[20]:


from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
import numpy as np

model_path = "./deberta_popularity_v3"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)

test_samples = [
    ("Top 10 Travel Destinations", "#travel #adventure", "Travel"),
    ("Funny Cat Compilation", "#funny #comedy #cats", "Comedy"),
    ("Cooking Recipe Tutorial", "#cooking #recipe", "Howto & Style")
]

print("Testing predictions:\n")
for title, hashtags, category in test_samples:
    label, probs = predict_popularity(title, hashtags, category, model, tokenizer)
    print(f"Title: {title}")
    print(f"Category: {category}")
    print(f"Predicted: {label}")
    print(f"Probabilities: Low={probs['Low']:.3f}, Medium={probs['Medium']:.3f}, High={probs['High']:.3f}\n")


# In[21]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
import torch

test_df = pd.read_csv("./data/test_data.csv")

# Use the same enhanced text building function as training
def build_combined_text(row):
    # Category ID factorization (category-aware modeling)
    category_id = int(row["category_id"]) if pd.notna(row["category_id"]) else -1
    category = str(row["category"]) if pd.notna(row["category"]) else "Unknown"
    
    # Structured features
    title_len = int(row["title_length"]) if pd.notna(row["title_length"]) else 0
    hashtag_count = int(row["hashtag_count"]) if pd.notna(row["hashtag_count"]) else 0
    duration_sec = float(row["duration_sec"]) if pd.notna(row["duration_sec"]) else 0.0
    log_duration = float(row["log_duration"]) if pd.notna(row["log_duration"]) else 0.0
    has_desc = 1 if pd.notna(row["description"]) and str(row["description"]).strip() != "" else 0
    
    # Text fields
    title = str(row["title"]) if pd.notna(row["title"]) else ""
    hashtags = str(row["hashtags"]) if pd.notna(row["hashtags"]) and str(row["hashtags"]).lower() != "nan" else ""
    
    # Combined text with structured features injected
    combined = (
        f"CATEGORY_ID: {category_id}. "
        f"CATEGORY: {category}. "
        f"TITLE_LENGTH: {title_len} words. "
        f"HASHTAG_COUNT: {hashtag_count}. "
        f"DURATION_SEC: {duration_sec:.1f}. "
        f"LOG_DURATION: {log_duration:.2f}. "
        f"HAS_DESCRIPTION: {has_desc}. "
        f"TITLE: {title}. "
        f"HASHTAGS: {hashtags}"
    )
    return combined

test_df["text"] = test_df.apply(build_combined_text, axis=1)

label_mapping = {"Low": 0, "Medium": 1, "High": 2}
test_df["label"] = test_df["popularity_label"].map(label_mapping)
test_df = test_df[test_df["label"].notna()]

test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

tokenizer = DebertaV2Tokenizer.from_pretrained("./deberta_popularity_v3")
model = DebertaV2ForSequenceClassification.from_pretrained("./deberta_popularity_v3")

test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

class YouTubeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

test_dataset = YouTubeDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir="./deberta_popularity_v3",
    per_device_eval_batch_size=32,
)

trainer = Trainer(
    model=model,
    args=training_args
)

predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

print("=" * 60)
print("Final Test Set Evaluation")
print("=" * 60)

accuracy = accuracy_score(test_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels, average="weighted", zero_division=0)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, pred_labels, target_names=["Low", "Medium", "High"], digits=4))


# In[29]:


# ============================================================
# Hybrid Regression Model: DeBERTa Embeddings + Structured Features
# ============================================================
# Extract embeddings from fine-tuned DeBERTa model
# Combine with structured features to predict log-transformed view growth
# Compare RandomForest vs LightGBM and select best model per target
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import joblib
import json
import os
import re

# Device selection: CUDA → MPS → CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

# Load data
print("\nLoading data...")
df = pd.read_csv("./data/train_data.csv")
print(f"  Original rows: {len(df)}")
print(f"  Unique videos: {df['video_id'].nunique()}")

# Drop rows with missing target values
target_cols = ["views_1d", "views_7d", "views_30d"]
df = df.dropna(subset=target_cols)
print(f"  After dropping missing targets: {len(df)} rows")


# In[30]:


# ============================================================
# Compute Missing Features
# ============================================================

def parse_duration_iso(duration_str):
    """Parse ISO 8601 duration format (PT39S, PT5M10S, PT1H2M3S) to seconds."""
    if pd.isna(duration_str) or duration_str == "":
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

print("\nComputing missing features...")

# Compute title_length if missing
if "title_length" not in df.columns:
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    print("  Computed: title_length")

# Compute hashtag_count if missing
if "hashtag_count" not in df.columns:
    df["hashtag_count"] = df["hashtags"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).lower() != "nan" else 0
    )
    print("  Computed: hashtag_count")

# Compute duration_sec if missing
if "duration_sec" not in df.columns and "duration" in df.columns:
    df["duration_sec"] = df["duration"].apply(parse_duration_iso)
    print("  Computed: duration_sec from duration")
elif "duration_sec" not in df.columns:
    df["duration_sec"] = 0.0
    print("  Warning: duration_sec not found, set to 0")

# Compute log_duration
df["log_duration"] = np.log1p(df["duration_sec"])
print("  Computed: log_duration = log1p(duration_sec)")

# Compute has_description
if "has_description" not in df.columns:
    df["has_description"] = df["description"].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0
    )
    print("  Computed: has_description")

# Compute category_id if missing
if "category_id" not in df.columns and "category" in df.columns:
    df["category_id"], _ = pd.factorize(df["category"])
    print("  Computed: category_id from category")
elif "category_id" not in df.columns:
    df["category_id"] = -1
    print("  Warning: category_id not found, set to -1")

# Prepare structured features
numeric_features = [
    "title_length",
    "hashtag_count",
    "duration_sec",
    "log_duration",
    "has_description",
    "category_id"
]

# Fill missing values
for col in numeric_features:
    if col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)

print(f"\nStructured features ready: {numeric_features}")


# In[31]:


# ============================================================
# Build Combined Text for DeBERTa (matching classification model format)
# ============================================================

def build_combined_text(row):
    """Build text input matching the classification model format."""
    category_id = int(row["category_id"]) if pd.notna(row["category_id"]) else -1
    category = str(row["category"]) if pd.notna(row["category"]) else "Unknown"
    title = str(row["title"]) if pd.notna(row["title"]) else ""
    hashtags = str(row["hashtags"]) if pd.notna(row["hashtags"]) and str(row["hashtags"]).lower() != "nan" else ""
    
    combined = f"CATEGORY_ID: {category_id}. CATEGORY: {category}. TITLE: {title}. HASHTAGS: {hashtags}"
    return combined

df["combined_text"] = df.apply(build_combined_text, axis=1)

# Prepare log-transformed targets
df["log_views_1d"] = np.log1p(df["views_1d"])
df["log_views_7d"] = np.log1p(df["views_7d"])
df["log_views_30d"] = np.log1p(df["views_30d"])

log_target_cols = ["log_views_1d", "log_views_7d", "log_views_30d"]
print(f"\nTarget columns (log-transformed): {log_target_cols}")

# Train/Test Split (90% train, 10% test)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"\nTrain set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")


# In[32]:


# ============================================================
# Load Fine-tuned DeBERTa Model and Extract Embeddings
# ============================================================

print("\nLoading fine-tuned DeBERTa model...")
model_path = "./deberta_popularity_v3"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
deberta_model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
deberta_model = deberta_model.to(device)
deberta_model.eval()
print(f"  Model loaded and moved to {device}")

# Dataset class for text encoding
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze() for k, v in encoding.items()}

def extract_embeddings(texts, model, tokenizer, device, batch_size=32):
    """Extract CLS embeddings from DeBERTa model."""
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get hidden states
            outputs = model.deberta(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract CLS token embedding (last layer, first token)
            cls_embeddings = hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    return np.vstack(embeddings)

print("\nExtracting DeBERTa embeddings...")
print("  Training set...")
train_embeddings = extract_embeddings(train_df["combined_text"], deberta_model, tokenizer, device)
print(f"    Shape: {train_embeddings.shape}")

print("  Test set...")
test_embeddings = extract_embeddings(test_df["combined_text"], deberta_model, tokenizer, device)
print(f"    Shape: {test_embeddings.shape}")

# Combine embeddings with structured features
print("\nCombining embeddings with structured features...")
train_numeric = train_df[numeric_features].values
test_numeric = test_df[numeric_features].values

X_train = np.hstack([train_embeddings, train_numeric])
X_test = np.hstack([test_embeddings, test_numeric])

print(f"  Train feature matrix: {X_train.shape}")
print(f"  Test feature matrix: {X_test.shape}")
print(f"    (DeBERTa embeddings: {train_embeddings.shape[1]} dims + {len(numeric_features)} structured features)")

# Prepare targets (log-transformed)
Y_train = train_df[log_target_cols].values
Y_test = test_df[log_target_cols].values
print(f"  Target matrix: {Y_train.shape}")


# In[33]:


# ============================================================
# MODEL 1: Random Forest Regressor (per target)
# ============================================================

print("\n" + "=" * 80)
print("Training Random Forest Regressors...")
print("=" * 80)

rf_models = {}
rf_results = {}

for i, target in enumerate(log_target_cols):
    print(f"\nTraining RF for {target}...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, Y_train[:, i])
    rf_models[target] = model
    
    # Predictions
    y_pred = model.predict(X_test)
    y_true = Y_test[:, i]
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    rf_results[target] = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }
    
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Compute averages
rf_results["avg"] = {
    "RMSE": np.mean([rf_results[t]["RMSE"] for t in log_target_cols]),
    "MAE": np.mean([rf_results[t]["MAE"] for t in log_target_cols]),
    "R²": np.mean([rf_results[t]["R²"] for t in log_target_cols])
}

print(f"\nRandom Forest Average: RMSE={rf_results['avg']['RMSE']:.4f}, "
      f"MAE={rf_results['avg']['MAE']:.4f}, R²={rf_results['avg']['R²']:.4f}")


# In[34]:


# ============================================================
# MODEL 2: LightGBM Regressor (per target)
# ============================================================

try:
    from lightgbm import LGBMRegressor
    
    print("\n" + "=" * 80)
    print("Training LightGBM Regressors...")
    print("=" * 80)
    
    lgb_models = {}
    lgb_results = {}
    
    for i, target in enumerate(log_target_cols):
        print(f"\nTraining LightGBM for {target}...")
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, Y_train[:, i])
        lgb_models[target] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_true = Y_test[:, i]
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        lgb_results[target] = {
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Compute averages
    lgb_results["avg"] = {
        "RMSE": np.mean([lgb_results[t]["RMSE"] for t in log_target_cols]),
        "MAE": np.mean([lgb_results[t]["MAE"] for t in log_target_cols]),
        "R²": np.mean([lgb_results[t]["R²"] for t in log_target_cols])
    }
    
    print(f"\nLightGBM Average: RMSE={lgb_results['avg']['RMSE']:.4f}, "
          f"MAE={lgb_results['avg']['MAE']:.4f}, R²={lgb_results['avg']['R²']:.4f}")
    
    lgb_available = True
    
except ImportError:
    print("\nLightGBM not available. Install with: pip install lightgbm")
    lgb_available = False
    lgb_models = {}
    lgb_results = {}


# In[35]:


# ============================================================
# Model Comparison and Selection (Best Model Per Target)
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

# Create comparison DataFrames
comparison_data = []

for target in log_target_cols:
    target_short = target.replace("log_views_", "")
    row = {
        "Target": target_short,
        "RF_RMSE": rf_results[target]["RMSE"],
        "RF_MAE": rf_results[target]["MAE"],
        "RF_R²": rf_results[target]["R²"]
    }
    if lgb_available:
        row.update({
            "LGB_RMSE": lgb_results[target]["RMSE"],
            "LGB_MAE": lgb_results[target]["MAE"],
            "LGB_R²": lgb_results[target]["R²"]
        })
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print("\nPer-Target Comparison:")
print(comparison_df.to_string(index=False))

# Select best model for each target (based on R²)
best_models = {}
best_model_names = {}

for target in log_target_cols:
    target_short = target.replace("log_views_", "")
    
    rf_r2 = rf_results[target]["R²"]
    if lgb_available:
        lgb_r2 = lgb_results[target]["R²"]
        if lgb_r2 > rf_r2:
            best_models[target] = lgb_models[target]
            best_model_names[target] = "lgb"
        else:
            best_models[target] = rf_models[target]
            best_model_names[target] = "rf"
    else:
        best_models[target] = rf_models[target]
        best_model_names[target] = "rf"
    
    print(f"\n{target_short}: Best model = {best_model_names[target].upper()} "
          f"(R² = {max(rf_r2, lgb_results[target]['R²'] if lgb_available else rf_r2):.4f})")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Random Forest - Avg R²: {rf_results['avg']['R²']:.4f}")
if lgb_available:
    print(f"LightGBM - Avg R²: {lgb_results['avg']['R²']:.4f}")


# In[36]:


# ============================================================
# Save Best Models Per Target
# ============================================================

os.makedirs("./models", exist_ok=True)

print("\nSaving best models...")

for target in log_target_cols:
    target_short = target.replace("log_views_", "")
    model_name = best_model_names[target]
    model_path = f"./models/{model_name}_{target_short}.pkl"
    
    joblib.dump(best_models[target], model_path)
    print(f"  ✅ {target_short}: {model_name.upper()} → {model_path}")

# Save metadata
metadata = {
    "embedding_dim": train_embeddings.shape[1],
    "numeric_features": numeric_features,
    "target_cols": log_target_cols,
    "best_models": {target.replace("log_views_", ""): best_model_names[target] for target in log_target_cols},
    "performance": {
        "rf_avg_r2": float(rf_results["avg"]["R²"]),
        "rf_avg_rmse": float(rf_results["avg"]["RMSE"]),
        "rf_avg_mae": float(rf_results["avg"]["MAE"])
    },
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test))
}

if lgb_available:
    metadata["performance"].update({
        "lgb_avg_r2": float(lgb_results["avg"]["R²"]),
        "lgb_avg_rmse": float(lgb_results["avg"]["RMSE"]),
        "lgb_avg_mae": float(lgb_results["avg"]["MAE"])
    })

metadata_path = "./models/growth_model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n✅ Metadata saved to: {metadata_path}")

print("\n" + "=" * 80)
print("Hybrid Regression Framework - Complete!")
print("=" * 80)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




