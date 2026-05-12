# News Archive NLP Pipeline (2010–2025)

An NLP pipeline for processing ~350K news headlines to extract entities, sentiment, and topics.

## Tech Stack
- pandas, pyarrow
- spaCy (NER)
- NLTK (sentiment)
- Gensim (topic modeling)
- pytest

## Key Outputs
- Named entity extraction
- Sentiment analysis
- Topic modeling (LDA)
- Temporal trend insights

## How to Run
```bash
git clone https://github.com/umerkniazi/news-nlp-pipeline.git
cd news-nlp-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm