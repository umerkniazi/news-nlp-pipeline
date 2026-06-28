import pandas as pd
from src.topic_modeler import TopicModeler

INPUT_PATH = "data/processed/processed_nlp_features.parquet"

def main():
    df = pd.read_parquet(INPUT_PATH)

    tm = TopicModeler()
    tm.fit(df["headline"].tolist())

    df = tm.assign_topics(df)

    tm.save_model()

    df.to_parquet(INPUT_PATH)

if __name__ == "__main__":
    main()