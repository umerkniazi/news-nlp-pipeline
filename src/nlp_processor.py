import spacy
from spacy.pipeline import EntityRuler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from src.config import SPACY_MODEL, SENTIMENT_THRESHOLD_POS, SENTIMENT_THRESHOLD_NEG, NER_OVERRIDES

class NLPProcessor:
    def __init__(self, spacy_model: str = SPACY_MODEL):
        self.nlp = spacy.load(spacy_model, disable=["parser", "tagger", "lemmatizer"])
        self.ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        patterns = []
        for term, label in NER_OVERRIDES.items():
            patterns.append({"label": label, "pattern": [{"LOWER": term}]})
        self.ruler.add_patterns(patterns)

        self.sia = SentimentIntensityAnalyzer()
        news_damping = {
            'blast': -0.5,
            'strike': -0.2,
            'ambush': -0.5,
            'kill': -0.8,
            'arrest': 0.0,
            'protest': -0.1
        }
        self.sia.lexicon.update(news_damping)
        
        self.valid_labels = {"PERSON", "ORG", "GPE", "LOC", "EVENT"}
        self.canonical_map = {
            "pti": "PTI", "pml-n": "PML-N", "ppp": "PPP", "cpec": "CPEC", 
            "ispr": "ISPR", "fbr": "FBR", "nab": "NAB", "sc": "SC", 
            "lhc": "LHC", "shc": "SHC", "ihc": "IHC", "phc": "PHC"
        }

    def clean_entity_text(self, text: str) -> str:
        txt = text.strip().rstrip(".,;:!?")
        if txt.lower().endswith("'s"):
            txt = txt[:-2]
        elif txt.lower().endswith("’s"):
            txt = txt[:-2]
        return txt.strip()

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, "neutral"
        score = self.sia.polarity_scores(text)['compound']
        if score >= SENTIMENT_THRESHOLD_POS:
            return score, "positive"
        if score <= SENTIMENT_THRESHOLD_NEG:
            return score, "negative"
        return score, "neutral"

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'headline', batch_size: int = 2000, n_process: int = 1) -> pd.DataFrame:
        texts = df[text_column].fillna("").astype(str).tolist()
        
        entities_list = []
        sentiment_scores = []
        sentiment_labels = []
        
        for doc in tqdm(self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts), desc="NER & Sentiment Processing"):
            row_ents = set()
            for ent in doc.ents:
                cleaned_text = self.clean_entity_text(ent.text)
                if not cleaned_text or len(cleaned_text) < 2:
                    continue
                    
                token_lower = cleaned_text.lower()
                if token_lower in NER_OVERRIDES:
                    label = NER_OVERRIDES[token_lower]
                    final_text = self.canonical_map.get(token_lower, cleaned_text.title())
                else:
                    label = ent.label_
                    final_text = self.canonical_map.get(token_lower, cleaned_text)
                    
                if label in self.valid_labels:
                    row_ents.add(f"{final_text} ({label})")
            
            entities_list.append(list(sorted(row_ents)))
            
            score, label_str = self.analyze_sentiment(doc.text)
            sentiment_scores.append(score)
            sentiment_labels.append(label_str)
            
        df_out = df.copy()
        df_out['extracted_entities'] = entities_list
        df_out['sentiment_score'] = sentiment_scores
        df_out['sentiment_label'] = sentiment_labels
        return df_out