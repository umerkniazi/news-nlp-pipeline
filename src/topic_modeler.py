import pandas as pd
import gensim
import json
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from pathlib import Path

from src.config import (
    NUM_TOPICS, 
    LDA_PASSES, 
    RANDOM_SEED, 
    LDA_MODEL_FILE, 
    DICTIONARY_FILE,
    DATA_DIR,
    TOPIC_MAPPING_FILE
)

class TopicModeler:
    def __init__(self, num_topics: int = NUM_TOPICS, passes: int = LDA_PASSES):
        self.num_topics = num_topics
        self.passes = passes
        self.dictionary = None
        self.lda_model = None

    def preprocess_text(self, texts: list[str]) -> list[list[str]]:
        custom_stops = STOPWORDS.union({
            'says', 'said', 'new', 'year', 'today', 'bn', 'rs', 'day', 'week', 
            'make', 'makes', 'want', 'wants', 'take', 'takes', 'according', 'set',
            'told', 'asked', 'report', 'reports', 'man', 'woman', 'people'
        })
        
        tokenized = [simple_preprocess(str(text), deacc=True) for text in texts]
        
        phrases = Phrases(tokenized, min_count=5, threshold=10)
        bigram_model = Phraser(phrases)
        
        processed = []
        for doc in bigram_model[tokenized]:
            filtered = [t for t in doc if t not in custom_stops and len(t) > 2]
            processed.append(filtered)
        return processed

    def fit(self, texts: list[str]) -> None:
        processed_docs = self.preprocess_text(texts)
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=5, no_above=0.3)
        
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        self.lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=RANDOM_SEED,
            passes=self.passes,
            workers=2,
            alpha='symmetric',
            eta='auto'
        )

    def assign_topics(self, df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
        if getattr(self, 'lda_model', None) is None:
            raise ValueError("Model not trained.")
            
        processed_docs = self.preprocess_text(df[text_column].tolist())
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        topics = []
        for bow in corpus:
            dist = self.lda_model.get_document_topics(bow)
            dominant_topic = max(dist, key=lambda x: x[1])[0] if dist else -1
            topics.append(dominant_topic)
            
        df_out = df.copy()
        df_out['dominant_topic'] = topics
        return df_out

    def assign_topic_labels(self, df: pd.DataFrame, mapping_path: Path = TOPIC_MAPPING_FILE) -> pd.DataFrame:
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            mapping = {int(k): v for k, v in mapping.items()}
            df['topic_name'] = df['dominant_topic'].map(mapping).fillna("General News")
        except FileNotFoundError:
            print(f"File NOT found at {mapping_path.resolve()}")
            df['topic_name'] = "Mapping Missing"
        return df

    def get_topic_descriptors(self, num_words: int = 15) -> dict:
        if self.lda_model is None:
            raise ValueError("Model not trained.")
        return {i: [w for w, p in self.lda_model.show_topic(i, topn=num_words)] 
                for i in range(self.num_topics)}

    def save_model(self, m_path: Path = LDA_MODEL_FILE, d_path: Path = DICTIONARY_FILE) -> None:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        self.lda_model.save(str(m_path))
        self.dictionary.save(str(d_path))

    def load_model(self, m_path: Path = LDA_MODEL_FILE, d_path: Path = DICTIONARY_FILE) -> None:
        self.lda_model = gensim.models.LdaMulticore.load(str(m_path))
        self.dictionary = corpora.Dictionary.load(str(d_path))