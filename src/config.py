import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "news_headlines.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_nlp_features.parquet"
NETWORK_DATA_FILE = PROCESSED_DATA_DIR / "entity_network.json"
LDA_MODEL_FILE = MODELS_DIR / "lda_topic_model.gensim"
DICTIONARY_FILE = MODELS_DIR / "corpus_dictionary.gensim"
TOPIC_MAPPING_FILE = DATA_DIR / "topic_mapping.json"

SPACY_MODEL = "en_core_web_sm"
SENTIMENT_THRESHOLD_POS = 0.05
SENTIMENT_THRESHOLD_NEG = -0.05

NUM_TOPICS = 20
LDA_PASSES = 50
RANDOM_SEED = 42
SAMPLE_SIZE = None

NER_OVERRIDES = {
    "brexit": "EVENT",
    "t20": "EVENT",
    "psl": "EVENT",
    "kurram": "GPE",
    "kpk": "GPE",
    "sri lanka": "GPE",
    "sri_lanka": "GPE",
    "covid": "EVENT",
    "twitter": "ORG"
}