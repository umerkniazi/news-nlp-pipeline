import pandas as pd
from src.topic_modeler import TopicModeler

def main():
    df = pd.read_parquet("data/processed/news.parquet")

    tm = TopicModeler()
    tm.fit(df["headline"].tolist())
    df = tm.assign_topics(df)
    tm.save_model()

    df.to_parquet("data/processed/topics_output.parquet")

if __name__ == "__main__":
    main()