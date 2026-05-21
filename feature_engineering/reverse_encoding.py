# Reverse binning with uniform distribution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
mean = 50
std_dev = 10
size = 1000
data = np.random.normal(mean, std_dev, size)
df = pd.DataFrame(data, columns=['Age'])

print(df.head())

n = len(df)
k = int(np.ceil(np.log2(n) + 1))

print(f"Number of bins (k): {k}")

from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform') # encode means giving values to the bins, ordinal means giving values in order, strategy means how to bin the data

df['Age_binned'] = discretizer.fit_transform(df[['Age']])

print(df.head())

sns.countplot(x = df['Age_binned'], data = df)