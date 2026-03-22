import pandas as pd

tr = pd.read_csv("training_data.csv")

print("=== SHAPE ===")
print(tr.shape)

print("\n=== TARGET ===")
print(tr["target"].value_counts())

print("\n=== SAMPLE ROWS ===")
print(tr.head(3).to_string())

print("\n=== DTYPES ===")
print(tr.dtypes)

print("\n=== NULLS ===")
print(tr.isnull().sum()[tr.isnull().sum() > 0])