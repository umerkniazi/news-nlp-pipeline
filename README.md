# news-nlp-pipeline

End-to-end NLP pipeline processing 350,718 Dawn News headlines (2010–2025) for topic modeling, named entity recognition, and sentiment analysis. No labels or supervision — the structure emerges from the data itself.

> **Note:** Built for academic research. Raw data not redistributed. All content remains property of [Dawn Media Group](https://www.dawn.com).

## What it found

- Cricket dominates Dawn's coverage — 40k+ headlines, nearly double the next topic (Energy Economy)
- Dawn skews negative: 36.2% negative, 22.6% positive, 41.2% neutral across 15 years
- US is the second most mentioned entity after Pakistan, ahead of India
- 404,583 named entities extracted across 20 discovered topics

## Stack

- **NLP:** spaCy (NER), NLTK VADER (sentiment)
- **Modeling:** BERTopic, sentence-transformers
- **Dashboard:** Streamlit, Plotly, PyVis
- **Data:** pandas, pyarrow

## Structure

```text
├── data/               # Processed Parquet files
├── notebooks/          # Research and EDA
├── scripts/            # Utility and precomputation scripts
├── src/                # Modular pipeline
├── app.py              # Streamlit dashboard
└── tests/              # Unit tests
```

## Run locally

```bash
git clone https://github.com/umerkniazi/news-nlp-pipeline.git
cd news-nlp-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/precompute_plots.py
streamlit run app.py
```