import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from typing import List, Tuple
from src.config import SPACY_MODEL, SENTIMENT_THRESHOLD_POS, SENTIMENT_THRESHOLD_NEG

nltk.download('vader_lexicon', quiet=True)

class NLPProcessor:
    def __init__(self, spacy_model: str = SPACY_MODEL):
        self.nlp = spacy.load(spacy_model, disable=["parser", "tagger", "lemmatizer"])
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, "neutral"
        score = self.sia.polarity_scores(text)['compound']
        if score >= SENTIMENT_THRESHOLD_POS:
            label = "positive"
        elif score <= SENTIMENT_THRESHOLD_NEG:
            label = "negative"
        else:
            label = "neutral"
        return score, label

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'headline', batch_size: int = 1000) -> pd.DataFrame:
        texts = df[text_column].fillna("").tolist()
        entities_list = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            ents = list(set([f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]))
            entities_list.append(ents)
        sentiments = [self.analyze_sentiment(text) for text in texts]
        df_out = df.copy()
        df_out['extracted_entities'] = entities_list
        df_out['sentiment_score'] = [s[0] for s in sentiments]
        df_out['sentiment_label'] = [s[1] for s in sentiments]
        return df_out