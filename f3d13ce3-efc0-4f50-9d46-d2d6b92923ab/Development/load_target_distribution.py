import pandas as pd

tr = pd.read_csv("training_data.csv")
print(tr["target"].value_counts())
print(tr["target"].value_counts(normalize=True).round(4))
