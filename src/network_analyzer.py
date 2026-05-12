import pandas as pd
import json
from itertools import combinations
from collections import Counter
from pathlib import Path
from src.config import NETWORK_DATA_FILE

class NetworkAnalyzer:
    def __init__(self, top_n_entities: int = 50):
        self.top_n_entities = top_n_entities

    def generate_network(self, df: pd.DataFrame, entity_column: str = 'extracted_entities') -> dict:
        all_entities = [ent for sublist in df[entity_column] for ent in sublist]
        entity_counts = Counter(all_entities)
        top_entities = [ent for ent, count in entity_counts.most_common(self.top_n_entities)]
        
        edges = []
        for entities in df[entity_column]:
            filtered = list(set([e for e in entities if e in top_entities]))
            if len(filtered) > 1:
                edges.extend(list(combinations(sorted(filtered), 2)))
        
        edge_counts = Counter(edges)
        
        nodes = [{"id": ent, "name": ent.split(' (')[0], "group": ent.split('(')[-1].replace(')', ''), "value": entity_counts[ent]} for ent in top_entities]
        links = [{"source": s, "target": t, "value": w} for (s, t), w in edge_counts.items()]
        
        return {"nodes": nodes, "links": links}

    def save_network(self, network_data: dict, file_path: Path = NETWORK_DATA_FILE) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(network_data, f, indent=4)