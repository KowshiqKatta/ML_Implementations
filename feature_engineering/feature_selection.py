# correlation coefficient technique

import pandas as pd
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
iris['species'] = label_encoder.fit_transform(iris['species'])

print(iris.head())

correlation_matrix = iris.corr()
print(correlation_matrix)

correlation_with_target = correlation_matrix['species'].abs().sort_values(ascending=False)

print(correlation_with_target)

selected_features = correlation_with_target[correlation_with_target > 0.5].index

print(selected_features)

new_df = iris[selected_features]
print(new_df.head())