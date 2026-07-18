import os
import json
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, "data", "processed", "processed_nlp_features.parquet")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "dashboard")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dashboard.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)
df.columns = df.columns.str.strip()

required_cols = [
    "published_at",
    "headline",
    "sentiment_label",
    "sentiment_score",
    "extracted_entities"
]

if "dominant_topic" in df.columns:
    required_cols.append("dominant_topic")
elif "topic_name" in df.columns:
    required_cols.append("topic_name")

df = df[[c for c in required_cols if c in df.columns]].copy()

df["date"] = pd.to_datetime(df["published_at"])
df["date_only"] = df["date"].dt.date

if "dominant_topic" in df.columns:
    try:
        mapping_path = os.path.join(ROOT_DIR, "data", "topic_mapping.json")
        with open(mapping_path, "r") as f:
            topic_mapping = json.load(f)
        df["topic_label"] = df["dominant_topic"].astype(str).map(topic_mapping)
        df["topic_label"] = df["topic_label"].fillna("Topic " + df["dominant_topic"].astype(str))
    except FileNotFoundError:
        df["topic_label"] = "Topic " + df["dominant_topic"].astype(str)
elif "topic_name" in df.columns:
    df["topic_label"] = df["topic_name"]

df.to_parquet(OUTPUT_PATH, index=False)