import pandas as pd

df = pd.read_csv("../../artifacts/ingestion/yellow_tripdata.csv")

df.info()
df.head()