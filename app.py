import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
import json

st.set_page_config(page_title="Dawn News Intelligence Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/processed_nlp_features.parquet")
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['published_at'])
    
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

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Loading Error: {e}")
    st.stop()

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

st.title("Dawn News Intelligence Analytics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headlines", f"{len(filtered_df):,}")
col2.metric("Average Sentiment", f"{filtered_df['sentiment_score'].mean():.3f}")
col3.metric("Active Topics", filtered_df['topic_label'].nunique())

all_entities = [ent for sublist in filtered_df['extracted_entities'].dropna() if isinstance(sublist, list) or isinstance(sublist, tuple) for ent in sublist]
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
    daily_vol = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index(name='count')
    fig_vol = px.line(daily_vol, x='date', y='count', title="Daily Headline Volume")
    fig_vol.update_traces(line_color='#1f77b4')
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Topic Frequency Over Time")
    topic_vol = filtered_df.groupby([filtered_df['date'].dt.date, 'topic_label']).size().reset_index(name='count')
    fig_topic_vol = px.area(topic_vol, x='date', y='count', color='topic_label', title="Stacked Topic Volume")
    st.plotly_chart(fig_topic_vol, use_container_width=True)

with tab2:
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(filtered_df, names='sentiment_label', hole=0.4, 
                         color='sentiment_label',
                         color_discrete_map={'positive':'#2ca02c', 'neutral':'#7f7f7f', 'negative':'#d62728'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_s2:
        st.subheader("Rolling Sentiment Average (7-Day)")
        daily_sent = filtered_df.groupby(filtered_df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sent['rolling_7d'] = daily_sent['sentiment_score'].rolling(7).mean()
        fig_sent_trend = px.line(daily_sent, x='date', y='rolling_7d')
        fig_sent_trend.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sent_trend, use_container_width=True)

with tab3:
    st.subheader("Top Extracted Entities")
    entity_counts = Counter(all_entities).most_common(20)
    ent_df = pd.DataFrame(entity_counts, columns=['Entity', 'Frequency'])
    
    fig_ent = px.bar(ent_df, x='Frequency', y='Entity', orientation='h', title="Most Mentioned Actors/Locations")
    fig_ent.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_ent, use_container_width=True)

    st.divider()

    st.subheader("Entity Keyword Profiler")
    
    top_50_entities = [e for e, c in Counter(all_entities).most_common(50)]
    selected_entity = st.selectbox("Select an Entity:", options=top_50_entities)
    
    if selected_entity:
        entity_mask = filtered_df['extracted_entities'].apply(
            lambda x: selected_entity in x if isinstance(x, (list, tuple)) else False
        )
        entity_specific_df = filtered_df[entity_mask]
        
        if len(entity_specific_df) > 0:
            vectorizer = CountVectorizer(stop_words='english', max_features=15)
            try:
                X = vectorizer.fit_transform(entity_specific_df['headline'])
                keywords = vectorizer.get_feature_names_out()
                counts = X.sum(axis=0).A1
                
                kw_df = pd.DataFrame({'Keyword': keywords, 'Frequency': counts}).sort_values(by='Frequency', ascending=False)
                
                fig_kw = px.bar(kw_df, x='Frequency', y='Keyword', orientation='h', 
                                title=f"Top Keywords for '{selected_entity}'",
                                color_discrete_sequence=['#ff7f0e'])
                fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_kw, use_container_width=True)
            except ValueError:
                st.info("Not enough text data to extract meaningful keywords for this entity.")

with tab4:
    st.subheader("Interactive Entity Co-occurrence Network")
    min_edges = st.slider("Minimum Co-occurrence Threshold", 2, 50, 5)
    
    @st.cache_data
    def build_network_graph(df_subset, threshold):
        edges = Counter()
        for entities in df_subset['extracted_entities'].dropna():
            if len(entities) > 1:
                clean_ents = [e.split(' (')[0] for e in entities]
                edges.update(combinations(sorted(clean_ents), 2))
                
        G = nx.Graph()
        for (u, v), weight in edges.items():
            if weight >= threshold:
                G.add_edge(u, v, weight=weight)
        return G

    G = build_network_graph(filtered_df, min_edges)
    
    if len(G.nodes) > 0:
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        net.repulsion(node_distance=150, spring_length=200)
        
        path = 'network.html'
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)
    else:
        st.warning("No network connections found at this threshold. Lower the slider or expand the date range.")

with tab5:
    st.subheader("Topic Modeling Profile")
    topic_dist = filtered_df['topic_label'].value_counts().reset_index()
    topic_dist.columns = ['Topic', 'Volume']
    fig_topics = px.bar(topic_dist, x='Volume', y='Topic', orientation='h', color='Topic')
    fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_topics, use_container_width=True)
    
    st.subheader("Representative Headlines per Topic")
    selected_topic = st.selectbox("Select Topic to View Sample Headlines", options=topic_dist['Topic'])
    sample_headlines = filtered_df[filtered_df['topic_label'] == selected_topic][['date', 'headline', 'sentiment_score']].head(10)
    st.dataframe(sample_headlines, use_container_width=True)

with tab6:
    st.subheader("Daily Headline Explorer")
    
    specific_day = st.date_input(
        "Select a Day:", 
        value=filtered_df['date'].max().date() if not filtered_df.empty else pd.to_datetime('today').date()
    )
    
    day_mask = filtered_df['date'].dt.date == specific_day
    day_df = filtered_df[day_mask]
    
    if len(day_df) > 0:
        st.metric(f"Headlines published on {specific_day}", len(day_df))
        display_df = day_df[['date', 'headline', 'topic_label', 'sentiment_label', 'sentiment_score']].sort_values(by='date')
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No headlines found for this specific date.")