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

# chi-square test

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

print(df)

discretizer = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
df_discretized = pd.DataFrame(discretizer.fit_transform(df.iloc[:, :-1]), columns = df.columns[:-1]) # chi square best works on categorical data, so we have to categorize the df

print(df_discretized)

df_discretized['target'] = df['target']

X = df_discretized.drop('target', axis = 1)
y = df_discretized['target']

chi2_selector = SelectKBest(chi2, k = 10)
X_kbest = chi2_selector.fit_transform(X, y)

selected_features = X.columns[chi2_selector.get_support()]
print(selected_features)

new_df = df[selected_features]
new_df['target'] = df['target']

print(new_df)

# ANOVA 

from sklearn.feature_selection import f_classif

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

print(df)

X = df.drop('target', axis = 1)
y = df['target']

anova_selector = SelectKBest(f_classif, k = 2) # select top 2 features
X_kbest = anova_selector.fit_transform(X, y) 

selected_features = X.columns[anova_selector.get_support()]

new_df = df[selected_features]
new_df['target'] = df['target']

print(new_df)

# Mutual information technique

from sklearn.feature_selection import mutual_info_classif, SelectKBest

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

X = df.drop('target', axis = 1)
y = df['target']

mi_selector = SelectKBest(mutual_info_classif, k = 10)
X_kbest = mi_selector.fit_transform(X, y)

selected_features = X.columns[mi_selector.get_support()]

new_df = df[selected_features]
new_df['target'] = df['target']

print(new_df)

# variance threshold technique

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

print(df)

variances = df.var()

print(variances)

from sklearn.feature_selection import VarianceThreshold

X = df.drop('target', axis = 1)
y = df['target']

selector = VarianceThreshold(threshold = 0.2)
X_transformed = selector.fit_transform(X)

selected_features = X.columns[selector.get_support()]

new_df = df[selected_features]
new_df['target'] = df['target']

print(new_df)