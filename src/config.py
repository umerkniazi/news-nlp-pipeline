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
    "t20": "OTHER",
    "psl": "EVENT",
    "ipl": "EVENT",
    "world cup": "EVENT",
    "t20 world cup": "EVENT",
    "covid": "EVENT",
    "covid-19": "EVENT",
    "coronavirus": "EVENT",
    "election": "OTHER",
    "elections": "OTHER",
    "champions trophy": "EVENT",
    "asia cup": "EVENT",
    "fifa world cup": "EVENT",
    "women’s world cup": "EVENT",
    "u-19 world cup": "EVENT",
    "french open": "EVENT",
    "wimbledon": "EVENT",
    "epl": "EVENT",
    "europa league": "EVENT",

    "pakistan": "GPE",
    "india": "GPE",
    "us": "GPE",
    "u.s.": "GPE",
    "usa": "GPE",
    "united states": "GPE",
    "uk": "GPE",
    "u.k.": "GPE",
    "united kingdom": "GPE",
    "china": "GPE",
    "sri lanka": "GPE",
    "sri_lanka": "GPE",
    "kpk": "GPE",
    "kp": "GPE",
    "kurram": "GPE",
    "karachi": "GPE",
    "lahore": "GPE",
    "islamabad": "GPE",
    "balochistan": "GPE",
    "punjab": "GPE",
    "sindh": "GPE",
    "gilgit-baltistan": "GPE",
    "multan": "GPE",
    "rawalpindi": "GPE",
    "quetta": "GPE",
    "peshawar": "GPE",
    "swabi": "GPE",
    "dir": "GPE",
    "panjgur": "GPE",
    "rahim yar khan": "GPE",
    "abbottabad": "GPE",

    "twitter": "ORG",
    "x": "ORG",
    "facebook": "ORG",
    "google": "ORG",
    "apple": "ORG",
    "microsoft": "ORG",
    "netflix": "ORG",
    "youtube": "ORG",
    "spacex": "ORG",
    "reuters": "ORG",

    "pti": "ORG",
    "pakistan tehreek-e-insaf": "ORG",
    "pml-n": "ORG",
    "pakistan muslim league-n": "ORG",
    "sbp": "ORG",
    "state bank": "ORG",
    "pak army": "ORG",
    "army": "ORG",
    "na": "ORG",
    "cec": "ORG",

    "imran": "PERSON",
    "imran khan": "PERSON",
    "nawaz": "PERSON",
    "nawaz sharif": "PERSON",
    "shehbaz sharif": "PERSON",
    "asif ali zardari": "PERSON",
    "zardari": "PERSON",
    "trump": "PERSON",
    "malala": "PERSON",
    "nisar": "PERSON",
    "maryam": "PERSON",
    "maryam nawaz": "PERSON",
    "qureshi": "PERSON",
    "altaf": "PERSON",
    "modi": "PERSON",
    "obama": "PERSON",
    "babar": "PERSON",
    "hamza": "PERSON",
    "gilani": "PERSON",
    "abbasi": "PERSON",
    "djokovic": "PERSON",
    "federer": "PERSON",
    "ronaldo": "PERSON",
    "neymar": "PERSON",
    "jinnah": "PERSON",

    "bahria": "ORG",
    "cpec": "ORG",
    "boko haram": "ORG",
    "houthi": "ORG",
    "jamaat": "ORG",
    "jamaat-i-islami": "ORG",

    "kashmiris": "OTHER",
    "kashmiri": "OTHER",
    "baloch": "OTHER",
    "rohingya": "OTHER",
    "urdu": "OTHER",
    "bollywood": "OTHER",
    "bill": "OTHER",
    "flour": "OTHER",
    "quran": "OTHER",
    "holy quran": "OTHER",
    "me too": "OTHER",
    "f1": "OTHER",

    "loC": "LOC",
    "loc": "LOC",
    "khyber": "LOC",
    "tirah": "LOC",
    "mediterranean": "LOC",
    "margalla hills": "LOC",
    "port qasim": "LOC",

    "u.s": "GPE",
    "usa.": "GPE",
    "uk.": "GPE"
}