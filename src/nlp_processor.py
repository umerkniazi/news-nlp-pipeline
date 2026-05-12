import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from src.config import SPACY_MODEL, SENTIMENT_THRESHOLD_POS, SENTIMENT_THRESHOLD_NEG, NER_OVERRIDES

nltk.download('vader_lexicon', quiet=True)

class NLPProcessor:
    def __init__(self, spacy_model: str = SPACY_MODEL):
        self.nlp = spacy.load(spacy_model, disable=["parser", "tagger", "lemmatizer"])
        self.sia = SentimentIntensityAnalyzer()
        self.valid_labels = {'PERSON', 'ORG', 'GPE'}

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, "neutral"
        score = self.sia.polarity_scores(text)['compound']
        if score >= SENTIMENT_THRESHOLD_POS:
            return score, "positive"
        if score <= SENTIMENT_THRESHOLD_NEG:
            return score, "negative"
        return score, "neutral"

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'headline', batch_size: int = 1000) -> pd.DataFrame:
        texts = df[text_column].fillna("").astype(str).tolist()
        entities_list = []
        
        for doc in tqdm(self.nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Extracting Entities"):
            row_ents = set()
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if not ent_text or len(ent_text) < 2:
                    continue
                    
                token_lower = ent_text.lower()
                if token_lower in NER_OVERRIDES:
                    label = NER_OVERRIDES[token_lower]
                else:
                    label = ent.label_
                    
                if label in self.valid_labels:
                    row_ents.add(f"{ent_text} ({label})")
            entities_list.append(list(row_ents))
            
        sentiments = [self.analyze_sentiment(text) for text in tqdm(texts, desc="Analyzing Sentiment")]
        df_out = df.copy()
        df_out['extracted_entities'] = entities_list
        df_out['sentiment_score'] = [s[0] for s in sentiments]
        df_out['sentiment_label'] = [s[1] for s in sentiments]
        return df_out