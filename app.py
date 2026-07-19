import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import json
import plotly.io as pio
import datetime

from utils.chart_style import apply_chart_style, SENTIMENT_COLORS

st.set_page_config(page_title="News NLP Pipeline | Dawn Analytics", page_icon="📰", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet("data/dashboard/dashboard.parquet")
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['published_at'])
    df['date_only'] = df['date'].dt.date

    if 'topic_label' not in df.columns:
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
def load_plot(path):
    try:
        return pio.read_json(path)
    except FileNotFoundError:
        return None


@st.cache_data
def load_network():
    try:
        with open("assets/plots/entity_network.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


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
def get_text_keywords(headlines_tuple, max_features=15):
    headlines = list(headlines_tuple)
    if not headlines:
        return pd.DataFrame(columns=['Keyword', 'Frequency'])
    try:
        vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
        X = vectorizer.fit_transform(headlines)
        keywords = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1
        return pd.DataFrame({'Keyword': keywords, 'Frequency': counts}).sort_values(by='Frequency', ascending=False)
    except ValueError:
        return pd.DataFrame(columns=['Keyword', 'Frequency'])


@st.cache_data
def dynamic_volume_plot(df):
    daily_volume = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily_volume.columns = ['date', 'count']
    fig = px.bar(
        daily_volume,
        x='date',
        y='count',
        title='Daily Headline Volume',
        labels={'date': 'Date', 'count': 'Number of Headlines'}
    )
    fig.update_layout(bargap=0.1)
    return apply_chart_style(fig)


@st.cache_data
def dynamic_topic_volume_plot(df):
    topic_daily = df.groupby([df['date'].dt.date, 'topic_label']).size().reset_index(name='count')
    topic_daily.columns = ['date', 'topic_label', 'count']
    fig = px.line(
        topic_daily,
        x='date',
        y='count',
        color='topic_label',
        title='Topic Frequency Over Time',
        labels={'date': 'Date', 'count': 'Headlines', 'topic_label': 'Topic'}
    )
    return apply_chart_style(fig)


@st.cache_data
def dynamic_sentiment_distribution(df):
    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        title='Sentiment Distribution',
        color='sentiment',
        color_discrete_map=SENTIMENT_COLORS
    )
    return apply_chart_style(fig)


@st.cache_data
def dynamic_topic_distribution(df):
    topic_dist = df['topic_label'].value_counts().reset_index()
    topic_dist.columns = ['topic', 'count']
    fig = px.bar(
        topic_dist,
        x='count',
        y='topic',
        orientation='h',
        title='Topic Distribution',
        labels={'count': 'Number of Headlines', 'topic': 'Topic'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    return apply_chart_style(fig)


@st.cache_data
def dynamic_rolling_sentiment(df):
    daily = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily.columns = ['date', 'sentiment_score']
    daily['rolling_7d'] = daily['sentiment_score'].rolling(7).mean()
    fig = px.line(
        daily,
        x='date',
        y='rolling_7d',
        title='7-Day Rolling Average Sentiment Score',
        labels={'date': 'Date', 'rolling_7d': '7-Day Rolling Average Sentiment Score'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return apply_chart_style(fig)


@st.cache_data
def get_topic_summary(df):
    topics = df['topic_label'].dropna().unique()
    summary = []
    for t in topics:
        t_df = df[df['topic_label'] == t]
        kw_df = get_text_keywords(tuple(t_df['headline'].tolist()), max_features=5)
        keywords = kw_df['Keyword'].tolist() if not kw_df.empty else []
        summary.append({
            'Topic': t,
            'Article Count': len(t_df),
            'Representative Keywords': ", ".join(keywords) if keywords else "N/A"
        })
    return pd.DataFrame(summary).sort_values(by='Article Count', ascending=False)


def format_entities_display(entities_list):
    if not isinstance(entities_list, (list, tuple, np.ndarray)) or len(entities_list) == 0:
        return ""
    clean_list = [str(e) for e in entities_list]
    if len(clean_list) > 3:
        return ", ".join(clean_list[:3]) + f" (+{len(clean_list) - 3} more)"
    return ", ".join(clean_list)


with st.spinner("Loading News NLP Pipeline dashboard..."):
    df = load_data()

st.sidebar.header("Dashboard Filters")
preset = st.sidebar.selectbox(
    "Date Range Preset",
    ["Full Dataset", "Last 7 Days", "Last 30 Days", "Last 3 Months", "Custom"]
)

max_date = df['date'].max().date()
min_date = df['date'].min().date()

if preset == "Last 7 Days":
    date_range = (max_date - datetime.timedelta(days=7), max_date)
elif preset == "Last 30 Days":
    date_range = (max_date - datetime.timedelta(days=30), max_date)
elif preset == "Last 3 Months":
    date_range = (max_date - datetime.timedelta(days=90), max_date)
elif preset == "Full Dataset":
    date_range = (min_date, max_date)
else:
    date_range = st.sidebar.date_input("Custom Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
    elif len(date_range) == 1:
        start_date, end_date = date_range[0], date_range[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = date_range, date_range

topic_filter = st.sidebar.multiselect("Filter by Topic", options=sorted(df['topic_label'].dropna().unique()))
sentiment_filter = st.sidebar.multiselect("Sentiment Category", options=['positive', 'neutral', 'negative'])
search_query = st.sidebar.text_input("Search Headlines")

mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)

if topic_filter:
    mask &= df['topic_label'].isin(topic_filter)
if sentiment_filter:
    mask &= df['sentiment_label'].isin(sentiment_filter)
if search_query:
    mask &= df['headline'].str.contains(search_query, case=False, na=False)

filtered_df = df[mask]
is_filtered = (preset != "Full Dataset") or bool(topic_filter) or bool(sentiment_filter) or bool(search_query)

st.title("Dawn News NLP Analytics")
st.markdown("**Part of the News NLP Pipeline project**")
st.markdown("An end-to-end NLP system for analyzing Dawn News articles using sentiment analysis, named entity recognition, and topic modeling.")

if filtered_df.empty:
    st.warning("No articles match your selected filters.")
    st.stop()

if is_filtered:
    st.info("Showing analytics generated from current filters.")
else:
    st.info("Showing corpus-level precomputed analytics.")

all_entities = get_all_entities(filtered_df)
top_entities_counts = get_entity_counts(all_entities, 1)
most_mentioned_entity = top_entities_counts[0][0] if top_entities_counts else "N/A"
most_common_topic = filtered_df['topic_label'].mode()[0] if not filtered_df['topic_label'].isnull().all() else "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Articles Analyzed", f"{len(filtered_df):,}")
col2.metric("Average Sentiment Score", f"{filtered_df['sentiment_score'].mean():.3f}")
col3.metric("Identified Topics", filtered_df['topic_label'].nunique())
col4.metric("Entity Mentions", f"{len(all_entities):,}")

col5, col6, col7 = st.columns(3)
col5.metric("Unique Entities", f"{len(set(all_entities)):,}")
col6.metric("Most Common Topic", most_common_topic)
col7.metric("Most Mentioned Entity", most_mentioned_entity)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Sentiment",
    "Entities",
    "Network",
    "Topics",
    "Model Insights",
    "Article Explorer"
])

with tab1:
    st.subheader("News Coverage Volume")
    if is_filtered:
        st.plotly_chart(dynamic_volume_plot(filtered_df), use_container_width=True, theme=None)
    else:
        fig_vol = load_plot("assets/plots/news_volume.json")
        if fig_vol is not None:
            st.plotly_chart(fig_vol, use_container_width=True, theme=None)
        else:
            st.error("Plot unavailable. Generate assets first.")

    st.subheader("Topic Frequency Over Time")
    if is_filtered:
        st.plotly_chart(dynamic_topic_volume_plot(filtered_df), use_container_width=True, theme=None)
    else:
        fig_topic_vol = load_plot("assets/plots/topic_volume.json")
        if fig_topic_vol is not None:
            st.plotly_chart(fig_topic_vol, use_container_width=True, theme=None)
        else:
            st.error("Plot unavailable. Generate assets first.")

with tab2:
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.subheader("Sentiment Distribution")
        if is_filtered:
            st.plotly_chart(dynamic_sentiment_distribution(filtered_df), use_container_width=True, theme=None)
        else:
            fig_pie = load_plot("assets/plots/sentiment_distribution.json")
            if fig_pie is not None:
                st.plotly_chart(fig_pie, use_container_width=True, theme=None)
            else:
                st.error("Plot unavailable. Generate assets first.")

    with col_s2:
        st.subheader("7-Day Rolling Average Sentiment Score")
        if is_filtered:
            st.plotly_chart(dynamic_rolling_sentiment(filtered_df), use_container_width=True, theme=None)
        else:
            fig_roll = load_plot("assets/plots/rolling_sentiment.json")
            if fig_roll is not None:
                st.plotly_chart(fig_roll, use_container_width=True, theme=None)
            else:
                st.error("Plot unavailable. Generate assets first.")

with tab3:
    st.subheader("Top Extracted Entities")
    top_50_entity_counts = get_entity_counts(all_entities, n=50)
    top_50_entities_list = [e for e, _ in top_50_entity_counts]

    if top_50_entity_counts:
        ent_df = pd.DataFrame(top_50_entity_counts[:20], columns=['Entity', 'Frequency'])
        fig_ent = px.bar(
            ent_df,
            x='Frequency',
            y='Entity',
            orientation='h',
            title='Top Extracted Entities',
            labels={'Frequency': 'Mentions'}
        )
        fig_ent.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(apply_chart_style(fig_ent), use_container_width=True, theme=None)

        st.divider()

        st.subheader("Entity Profile Explorer")
        selected_entity = st.selectbox("Select an Entity:", options=top_50_entities_list)

        if selected_entity:
            entity_mask = filtered_df['extracted_entities'].apply(
                lambda x: selected_entity in x if isinstance(x, (list, tuple, np.ndarray)) else False
            )
            entity_specific_df = filtered_df[entity_mask]
            mentions = sum([1 for e in all_entities if e == selected_entity])

            ec1, ec2 = st.columns(2)
            ec1.metric("Total Mentions", f"{mentions:,}")
            ec2.metric("Articles Containing Entity", f"{len(entity_specific_df):,}")

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.markdown("##### Common Topics")
                topic_counts = entity_specific_df['topic_label'].value_counts()
                for t, c in topic_counts.head(5).items():
                    pct = (c / len(entity_specific_df)) * 100
                    st.write(f"- **{t}**: {pct:.1f}% ({c} articles)")

                st.markdown("##### Top Associated Keywords")
                kw_df = get_text_keywords(tuple(entity_specific_df['headline'].tolist()))
                if not kw_df.empty:
                    fig_kw = px.bar(
                        kw_df.head(10),
                        x='Frequency',
                        y='Keyword',
                        orientation='h',
                        title='Top Associated Keywords',
                        labels={'Frequency': 'Co-occurrence Count'}
                    )
                    fig_kw.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(apply_chart_style(fig_kw), use_container_width=True, theme=None)
                else:
                    st.info("Not enough text data to extract keywords.")
            with col_p2:
                st.markdown("##### Sentiment Distribution")
                st.plotly_chart(dynamic_sentiment_distribution(entity_specific_df), use_container_width=True, theme=None)
    else:
        st.info("No entities identified within the filtered corpus.")

with tab4:
    st.subheader("Entity Co-occurrence Network")
    st.caption("Entity network represents relationships across the complete corpus.")

    network_html = load_network()
    if network_html is not None:
        components.html(network_html, height=650)
    else:
        st.error("Entity network not generated.")

with tab5:
    st.subheader("Discovered News Themes")
    if is_filtered:
        st.plotly_chart(dynamic_topic_distribution(filtered_df), use_container_width=True, theme=None)
    else:
        fig_topics = load_plot("assets/plots/topic_distribution.json")
        if fig_topics is not None:
            st.plotly_chart(fig_topics, use_container_width=True, theme=None)
        else:
            st.error("Plot unavailable. Generate assets first.")

    st.subheader("Topic Profile Explorer")
    topic_dist = filtered_df['topic_label'].value_counts().reset_index()
    topic_dist.columns = ['Topic', 'Volume']

    selected_topic = st.selectbox("Select Topic to Explore", options=topic_dist['Topic'])
    if selected_topic:
        t_df = filtered_df[filtered_df['topic_label'] == selected_topic]

        tc1, tc2 = st.columns([1, 1])
        with tc1:
            st.markdown("##### Top Keywords")
            kw_df = get_text_keywords(tuple(t_df['headline'].tolist()), max_features=10)
            if not kw_df.empty:
                st.write(", ".join(kw_df['Keyword'].tolist()))
            else:
                st.write("Not enough text data to extract keywords.")

            st.markdown("##### Topic Metric Profile")
            st.metric("Articles in Topic", f"{len(t_df):,}")
        with tc2:
            st.markdown("##### Sentiment Distribution")
            st.plotly_chart(dynamic_sentiment_distribution(t_df), use_container_width=True, theme=None)

        st.markdown("##### Extracted Entities within Theme")
        t_entities = get_all_entities(t_df)
        t_ent_counts = get_entity_counts(t_entities, n=10)
        if t_ent_counts:
            t_ent_df = pd.DataFrame(t_ent_counts, columns=['Entity', 'Mentions'])
            st.dataframe(t_ent_df, use_container_width=True, hide_index=True)

        st.markdown("##### Sample Headlines")
        sample_headlines = t_df[['date', 'headline', 'sentiment_label', 'sentiment_score', 'extracted_entities']].sample(min(10, len(t_df)), random_state=42).copy()
        sample_headlines['extracted_entities'] = sample_headlines['extracted_entities'].apply(format_entities_display)
        sample_headlines.columns = ['Date', 'Headline', 'Sentiment Label', 'Score', 'Extracted Entities']
        st.dataframe(sample_headlines, use_container_width=True, hide_index=True)

with tab6:
    st.subheader("Model Insights")
    st.write("Transparency into NLP pipeline outputs for model evaluation.")

    st.markdown("#### A) Sentiment Examples")
    st.dataframe(filtered_df[['headline', 'sentiment_label', 'sentiment_score']].sample(min(10, len(filtered_df)), random_state=42), use_container_width=True, hide_index=True)

    st.markdown("#### B) Named Entity Recognition Examples")
    ner_sample = filtered_df[['headline', 'extracted_entities']].sample(min(10, len(filtered_df)), random_state=42).copy()
    ner_sample['extracted_entities'] = ner_sample['extracted_entities'].apply(format_entities_display)
    st.dataframe(ner_sample, use_container_width=True, hide_index=True)

    st.markdown("#### C) Topic Examples")
    topic_summary = get_topic_summary(filtered_df)
    st.dataframe(topic_summary, use_container_width=True, hide_index=True)

with tab7:
    st.subheader("Article Explorer")

    specific_day = st.date_input(
        "Explore articles from a specific day",
        value=filtered_df['date'].max().date() if not filtered_df.empty else datetime.date.today()
    )

    day_df = filtered_df[filtered_df['date_only'] == specific_day].sort_values(by='date')

    if len(day_df) > 0:
        page_size = 50
        total_pages = max(1, (len(day_df) - 1) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * page_size
        st.caption(f"Showing {start + 1}–{min(start + page_size, len(day_df))} of {len(day_df)} headlines")

        display_df = day_df.iloc[start:start + page_size].copy()
        display_df['extracted_entities'] = display_df['extracted_entities'].apply(format_entities_display)
        display_df = display_df[['date', 'headline', 'topic_label', 'sentiment_label', 'sentiment_score', 'extracted_entities']]
        display_df.columns = ['Date', 'Headline', 'Topic Theme', 'Sentiment', 'Score', 'Identified Entities']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No articles found for this date.")

st.divider()
with st.expander("Methodology"):
    st.markdown("""
    * **Preprocessing**: Text normalization and cleaning of raw news headlines.
    * **Sentiment Classification & Scoring**: Transformer-based classification and scoring of textual sentiment.
    * **Named Entity Recognition (NER)**: Extraction of key people, locations, and organizations.
    * **Topic Modeling**: Unsupervised clustering of news themes using BERTopic.
    * **Entity Co-occurrence Network**: Graph-based analysis of entities frequently appearing in the same context.
    """)

st.caption("News NLP Pipeline | Built with Python, Streamlit, Transformers, NER, and BERTopic")