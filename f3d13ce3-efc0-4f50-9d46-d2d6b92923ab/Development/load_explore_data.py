
tr = pd.read_csv("training_data.csv")
te = pd.read_csv("test_data_hackathon.csv")

print(tr.shape, te.shape)
print(tr["target"].value_counts(normalize=True).round(3))