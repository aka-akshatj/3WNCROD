from scipy.io import loadmat

data = loadmat("Datasets/annthyroid.mat")

X = data["trandata"]

print(X[:10])      # first 10 rows
print(X.shape)

import pandas as pd

df = pd.DataFrame(X)
print(df.head())

print(df.describe())