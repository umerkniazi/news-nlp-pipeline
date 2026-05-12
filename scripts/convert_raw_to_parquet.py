import pandas as pd

df = pd.read_csv("news_headlines.csv")
df.to_parquet("news_headlines.parquet", index=False)