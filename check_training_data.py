import pandas as pd

df = pd.read_parquet("data/gold/training_dataset.parquet")

print("=== Shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Click label distribution ===")
print(df["clicked"].value_counts(dropna=False))

print("\n=== Sample rows ===")
print(df.head(10))

print("\n=== Null counts (top 20) ===")
print(df.isnull().sum().sort_values(ascending=False).head(20))