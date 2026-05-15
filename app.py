import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import json
import plotly.io as pio

st.set_page_config(page_title="Dawn News Intelligence Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/processed_nlp_features.parquet")
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['published_at'])
    df['date_only'] = df['date'].dt.date

    if 'dominant_topic' in df.columns:
        try:
            with open("data/topic_mapping.json", "r") as f:
                topic_mapping = json.load(f)
            df['topic_label'] = df['dominant_topic'].astype(str).map(topic_mapping)
            df['topic_label'] = df['topic_label'].fillna("Topic " + df['dominant_topic'].astype(str))
        except FileNotFoundError:
            df['topic_label'] = "Topic " + df['dominant_topic'].astype(str)
    elif 'topic_name' in df.columns:
        df['topic_label'] = df['topic_name']
    else:
        st.error(f"CRITICAL: No topic column found. Columns found: {df.columns.tolist()}")
        st.stop()

    return df

@st.cache_data
def get_all_entities(df):
    return [
        ent for sublist in df['extracted_entities'].dropna()
        if isinstance(sublist, (list, tuple, np.ndarray)) and len(sublist) > 0
        for ent in sublist
    ]

@st.cache_data
def get_entity_counts(entity_list, n=50):
    return Counter(entity_list).most_common(n)

@st.cache_data
def get_daily_sentiment(df):
    daily = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily['rolling_7d'] = daily['sentiment_score'].rolling(7).mean()
    return daily

@st.cache_data
def get_entity_keywords(headlines_tuple):
    headlines = list(headlines_tuple)
    vectorizer = CountVectorizer(stop_words='english', max_features=15)
    X = vectorizer.fit_transform(headlines)
    keywords = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    return pd.DataFrame({'Keyword': keywords, 'Frequency': counts}).sort_values(by='Frequency', ascending=False)

df = load_data()

st.sidebar.header("Dashboard Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    [df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

topic_filter = st.sidebar.multiselect("Filter by Topic", options=df['topic_label'].dropna().unique())
sentiment_filter = st.sidebar.multiselect("Sentiment Category", options=['positive', 'neutral', 'negative'])
search_query = st.sidebar.text_input("Search Headlines")

mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])

if topic_filter:
    mask &= df['topic_label'].isin(topic_filter)
if sentiment_filter:
    mask &= df['sentiment_label'].isin(sentiment_filter)
if search_query:
    mask &= df['headline'].str.contains(search_query, case=False, na=False)

filtered_df = df[mask]

all_entities = get_all_entities(filtered_df)
top_50_entity_counts = get_entity_counts(all_entities, n=50)
top_20_entities = top_50_entity_counts[:20]
top_50_entities = [e for e, _ in top_50_entity_counts]

st.title("Dawn News Intelligence Analytics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headlines", f"{len(filtered_df):,}")
col2.metric("Average Sentiment", f"{filtered_df['sentiment_score'].mean():.3f}")
col3.metric("Active Topics", filtered_df['topic_label'].nunique())
col4.metric("Total Entities Extracted", f"{len(all_entities):,}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Temporal & Volume",
    "🎭 Sentiment Analysis",
    "🏛️ Entity Analytics",
    "🕸️ Entity Network",
    "📑 Topic Modeling",
    "📅 Daily Deep Dive"
])

with tab1:
    st.subheader("Headline Volume Over Time")
    fig_vol = pio.read_json("assets/plots/news_volume.json")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Topic Frequency Over Time")
    fig_topic_vol = pio.read_json("assets/plots/topic_volume.json")
    st.plotly_chart(fig_topic_vol, use_container_width=True)

with tab2:
    col_s1, col_s2 = st.columns([1, 2])

    with col_s1:
        st.subheader("Sentiment Distribution")
        fig_pie = pio.read_json("assets/plots/sentiment_distribution.json")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        st.subheader("Rolling Sentiment Average (7-Day)")
        daily_sent = get_daily_sentiment(filtered_df)
        fig_sent_trend = px.line(daily_sent, x='date', y='rolling_7d')
        fig_sent_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sent_trend, use_container_width=True)

with tab3:
    st.subheader("Top Extracted Entities")

    ent_df = pd.DataFrame(top_20_entities, columns=['Entity', 'Frequency'])
    fig_ent = px.bar(ent_df, x='Frequency', y='Entity', orientation='h')
    fig_ent.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_ent, use_container_width=True)

    st.divider()

    st.subheader("Entity Keyword Profiler")

    selected_entity = st.selectbox("Select an Entity:", options=top_50_entities)

    if selected_entity:
        entity_mask = filtered_df['extracted_entities'].apply(
            lambda x: selected_entity in x if isinstance(x, (list, tuple, np.ndarray)) else False
        )
        entity_specific_df = filtered_df[entity_mask]

        if len(entity_specific_df) > 0:
            try:
                kw_df = get_entity_keywords(tuple(entity_specific_df['headline'].tolist()))
                fig_kw = px.bar(kw_df, x='Frequency', y='Keyword', orientation='h')
                fig_kw.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_kw, use_container_width=True)
            except ValueError:
                st.info("Not enough text data to extract keywords.")

with tab4:
    st.subheader("Entity Co-occurrence Network")
    st.caption("Top 80 entities · minimum 5 co-occurrences · computed across full dataset")

    try:
        with open("assets/plots/entity_network.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=650)
    except FileNotFoundError:
        st.warning("Network not found. Run scripts/precompute_plots.py to generate it.")

with tab5:
    st.subheader("Topic Modeling Profile")

    fig_topics = pio.read_json("assets/plots/topic_distribution.json")
    st.plotly_chart(fig_topics, use_container_width=True)

    st.subheader("Representative Headlines per Topic")

    topic_dist = filtered_df['topic_label'].value_counts().reset_index()
    topic_dist.columns = ['Topic', 'Volume']

    selected_topic = st.selectbox("Select Topic to View Sample Headlines", options=topic_dist['Topic'])
    sample_headlines = filtered_df[filtered_df['topic_label'] == selected_topic][['date', 'headline', 'sentiment_score']].head(10)
    st.dataframe(sample_headlines, use_container_width=True)

with tab6:
    st.subheader("Daily Headline Explorer")

    specific_day = st.date_input(
        "Select a Day:",
        value=filtered_df['date'].max().date() if not filtered_df.empty else pd.to_datetime('today').date()
    )

    day_df = filtered_df[filtered_df['date_only'] == specific_day][['date', 'headline', 'topic_label', 'sentiment_label', 'sentiment_score']].sort_values(by='date')

    if len(day_df) > 0:
        st.metric(f"Headlines published on {specific_day}", len(day_df))

        page_size = 50
        total_pages = max(1, (len(day_df) - 1) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * page_size
        st.caption(f"Showing {start + 1}–{min(start + page_size, len(day_df))} of {len(day_df)} headlines")
        st.dataframe(day_df.iloc[start:start + page_size], use_container_width=True, hide_index=True)
    else:
        st.info("No headlines found for this specific date.")