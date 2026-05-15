import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import os
import numpy as np
from collections import Counter
from itertools import combinations
from pyvis.network import Network

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "assets", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(os.path.join(ROOT_DIR, "data", "processed", "processed_nlp_features.parquet"))
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['published_at'])

try:
    with open(os.path.join(ROOT_DIR, "data", "topic_mapping.json"), "r") as f:
        topic_mapping = json.load(f)
    df['topic_label'] = df['dominant_topic'].astype(str).map(topic_mapping)
    df['topic_label'] = df['topic_label'].fillna("Topic " + df['dominant_topic'].astype(str))
except FileNotFoundError:
    df['topic_label'] = "Topic " + df['dominant_topic'].astype(str)

daily_volume = df.groupby(df['date'].dt.date).size().reset_index(name='count')
daily_volume.columns = ['date', 'count']
fig_vol = px.bar(
    daily_volume,
    x='date',
    y='count',
    title='Daily Headline Volume',
    labels={'date': 'Date', 'count': 'Number of Headlines'}
)
fig_vol.update_layout(bargap=0.1)
pio.write_json(fig_vol, os.path.join(OUTPUT_DIR, "news_volume.json"))
print("Saved news_volume.json")

topic_daily = df.groupby([df['date'].dt.date, 'topic_label']).size().reset_index(name='count')
topic_daily.columns = ['date', 'topic_label', 'count']
fig_topic_vol = px.line(
    topic_daily,
    x='date',
    y='count',
    color='topic_label',
    title='Topic Frequency Over Time',
    labels={'date': 'Date', 'count': 'Headlines', 'topic_label': 'Topic'}
)
pio.write_json(fig_topic_vol, os.path.join(OUTPUT_DIR, "topic_volume.json"))
print("Saved topic_volume.json")

sentiment_counts = df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']
color_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
fig_pie = px.pie(
    sentiment_counts,
    names='sentiment',
    values='count',
    title='Sentiment Distribution',
    color='sentiment',
    color_discrete_map=color_map
)
pio.write_json(fig_pie, os.path.join(OUTPUT_DIR, "sentiment_distribution.json"))
print("Saved sentiment_distribution.json")

topic_dist = df['topic_label'].value_counts().reset_index()
topic_dist.columns = ['topic', 'count']
fig_topics = px.bar(
    topic_dist,
    x='count',
    y='topic',
    orientation='h',
    title='Topic Distribution',
    labels={'count': 'Number of Headlines', 'topic': 'Topic'}
)
fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
pio.write_json(fig_topics, os.path.join(OUTPUT_DIR, "topic_distribution.json"))
print("Saved topic_distribution.json")


entity_counter = Counter()
edges = Counter()
for entities in df['extracted_entities'].dropna():
    if isinstance(entities, (list, tuple, np.ndarray)) and len(entities) > 1:
        clean = [e.split(' (')[0] for e in entities]
        entity_counter.update(clean)
        edges.update(combinations(sorted(clean), 2))

top_entities = {e for e, _ in entity_counter.most_common(80)}
top_edges = [(u, v, w) for (u, v), w in edges.most_common(200)
             if u in top_entities and v in top_entities]

added_nodes = set()
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
for u, v, w in top_edges:
    if u not in added_nodes:
        net.add_node(u, label=u)
        added_nodes.add(u)
    if v not in added_nodes:
        net.add_node(v, label=v)
        added_nodes.add(v)
    net.add_edge(u, v, value=w)

net.repulsion(node_distance=150, spring_length=200)
network_path = os.path.join(OUTPUT_DIR, "entity_network.html")
net.save_graph(network_path)
print(f"Saved entity_network.html ({len(net.nodes)} nodes, {len(net.edges)} edges)")

print("\nAll plots precomputed and saved to", OUTPUT_DIR)