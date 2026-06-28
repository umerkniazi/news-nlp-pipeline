import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.config import (
    NUM_TOPICS,
    RANDOM_SEED,
    LDA_MODEL_FILE,
    TOPIC_MAPPING_FILE,
)

BERTOPIC_MODEL_FILE = LDA_MODEL_FILE.parent / "bertopic_model"


class TopicModeler:
    def __init__(self, num_topics: int = NUM_TOPICS):
        self.num_topics = num_topics
        self.model = None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, texts: list[str]) -> None:
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2),
        )

        umap_model = UMAP(
            n_neighbors=10,
            n_components=3,
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_SEED,
        )

        cluster_model = MiniBatchKMeans(
            n_clusters=self.num_topics,
            batch_size=1024,
            random_state=RANDOM_SEED,
            n_init="auto",
        )

        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer,
            calculate_probabilities=False,
            verbose=True,
            language="english",
        )

        print("Computing embeddings...")

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=16,
        ).astype(np.float32)

        print("Fitting BERTopic...")

        self.model.fit(texts, embeddings=embeddings)

        del embeddings
        gc.collect()

    def assign_topics(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")

        topics, _ = self.model.transform(df[text_column].tolist())

        df_out = df.copy()
        df_out["dominant_topic"] = topics

        return df_out

    def assign_topic_labels(
        self,
        df: pd.DataFrame,
        mapping_path: Path = TOPIC_MAPPING_FILE,
    ) -> pd.DataFrame:
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)

            mapping = {int(k): v for k, v in mapping.items()}

            df["topic_name"] = (
                df["dominant_topic"]
                .map(mapping)
                .fillna("General News")
            )

        except FileNotFoundError:
            print(f"File NOT found at {mapping_path.resolve()}")
            df["topic_name"] = "Mapping Missing"

        return df

    def get_topic_descriptors(self, num_words: int = 15) -> dict:
        if self.model is None:
            raise ValueError("Model not trained.")

        descriptors = {}

        for topic_id in self.model.get_topics():
            if topic_id == -1:
                continue

            words = [
                word
                for word, _ in self.model.get_topic(topic_id)[:num_words]
            ]

            descriptors[topic_id] = words

        return descriptors

    def save_model(
        self,
        m_path: Path = BERTOPIC_MODEL_FILE,
        **kwargs,
    ) -> None:
        m_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(
            str(m_path),
            serialization="safetensors",
            save_embedding_model=False,
        )

        print(f"Model saved to {m_path}")

    def load_model(
        self,
        m_path: Path = BERTOPIC_MODEL_FILE,
        **kwargs,
    ) -> None:
        self.model = BERTopic.load(
            str(m_path),
            embedding_model=self.embedding_model,
        )

        print(f"Model loaded from {m_path}")