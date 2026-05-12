import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_FILE, PROCESSED_DATA_FILE, SAMPLE_SIZE

def load_raw_data(file_path: Path = RAW_DATA_FILE, sample_size: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    """Loads raw CSV data, drops nulls in critical columns, and optionally samples."""
    df = pd.read_csv(file_path)
    
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['headline', 'category']).copy()
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    return df

def save_processed_data(df: pd.DataFrame, file_path: Path = PROCESSED_DATA_FILE) -> None:
    """Saves the DataFrame to Parquet format, ensuring the directory exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)

def load_processed_data(file_path: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    """Loads processed Parquet data."""
    return pd.read_parquet(file_path)