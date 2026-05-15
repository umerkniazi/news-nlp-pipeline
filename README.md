# Pakistan News Intelligence NLP Pipeline (2010–2025)
> **⚠️ Legal & Ethical Notice:** This repository is for **academic research purposes only**. The data collection modules (scrapers) are designed with respectful rate-limiting and custom headers to adhere to ethical crawling standards and avoid server strain. All news content and headlines remain the intellectual property of the [Dawn Media Group](https://www.dawn.com). This project does not redistribute the raw news dataset; it provides the framework for processing and analysis.

An end-to-end NLP pipeline designed to process ~350,000 news headlines from Dawn News to extract entities, sentiment, and latent topics.

## Tech Stack
- **Data:** pandas, pyarrow (Parquet)
- **NLP:** spaCy (NER), NLTK (VADER Sentiment)
- **Modeling:** Gensim (LDA Topic Modeling)
- **Dashboard:** Streamlit, Plotly, PyVis

## Key Outputs
- **Named Entity Recognition:** Identifying key political figures, locations, and organizations.
- **Sentiment Analysis:** Tracking national mood and reporting tone over 15 years.
- **Topic Modeling:** Discovering hidden themes in regional reporting (Politics, Economy, Security).
- **Temporal Insights:** Visualizing how news volume and sentiment shift across different events.

## Project Structure
```text
├── data/               # Raw and processed Parquet files
├── notebooks/          # Research and EDA notebooks
├── scripts/            # Utility scripts (e.g. plot precomputation)
├── src/                # Modular NLP pipeline scripts
├── app.py              # Streamlit dashboard
└── tests/              # Unit tests for NLP logic
```

## How to Run
```bash
git clone https://github.com/umerkniazi/news-nlp-pipeline.git
cd news-nlp-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the Pipeline
```bash
python src/run_pipeline.py
```

### Precompute Dashboard Plots
Run this once after the pipeline completes, and again whenever the underlying data changes:
```bash
python scripts/precompute_plots.py
```
This generates static Plotly charts and the entity co-occurrence network into `assets/plots/`, so the dashboard loads instantly without recomputing them at runtime.

### Launch the Analytics Dashboard
```bash
streamlit run app.py
```