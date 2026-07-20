# News NLP Pipeline

End-to-end NLP pipeline analyzing 350,718 Dawn News headlines (2010–2025) using topic modeling, named entity recognition and sentiment analysis. The dataset contains no labels or supervision. The structure emerges entirely from the data itself.

> **Note:** Built for academic research. Raw data is not redistributed. All content remains the property of [Dawn Media Group](https://www.dawn.com).

## What it found

- Cricket is the most frequently discussed topic, with approximately 32.5k headlines, followed by Legal Cases with approximately 29.2k headlines
- The dataset has an average sentiment score of **-0.085**, indicating a slight overall negative sentiment
- Pakistan is the most frequently mentioned entity, with the United States ranking second ahead of India
- 420,587 named entity mentions were extracted, representing 45,249 unique entities across 20 discovered topics

These findings are exploratory rather than definitive. They demonstrate how combining multiple NLP techniques can reveal patterns that would be difficult to identify through any single method alone.

## Stack

- **NLP:** spaCy (NER), NLTK VADER (sentiment)
- **Modeling:** BERTopic, sentence-transformers
- **Dashboard:** Streamlit, Plotly, PyVis
- **Data:** pandas, pyarrow

## Project Structure

```text
├── data/               # Processed Parquet files
├── notebooks/          # Research and exploratory analysis
├── scripts/            # Utility and preprocessing scripts
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
python -m nltk.downloader vader_lexicon
python scripts/precompute_plots.py
streamlit run app.py
```