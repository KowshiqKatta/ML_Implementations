# mean imputation

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    'Value': [1, 2, np.nan, 4, 5, np.nan, 7]
}

df = pd.DataFrame(data)
print(df.head())

imputer = SimpleImputer(strategy='mean')
df['transformed_value'] = imputer.fit_transform(df[['Value']])

print(df)

# median imputation
imputer = SimpleImputer(strategy='median')
df['transformed_value'] = imputer.fit_transform(df[['Value']])

print(df)

# mode imputation
imputer = SimpleImputer(strategy='most_frequent')
df['transformed_value'] = imputer.fit_transform(df[['Value']])

print(df)

# constant imputation

imputer = SimpleImputer(strategy='constant', fill_value=0)
df['transformed_value'] = imputer.fit_transform(df[['Value']])

print(df)

# forward fill imputation

df['transformed_value'] = df['Value'].ffill()
print(df)

# backward fill imputation
df['transformed_value'] = df['Value'].bfill()
print(df)

# interpolation imputation

df['transformed_value'] = df['Value'].interpolate(method='linear')
print(df)

# moving average imputation

df['transformed_value'] = df['Value'].fillna(df['Value'].rolling(window=3, min_periods=1).mean())
print(df)

# end of distribution imputation

P_low = df['Value'].quantile(0.1)
P_high = df['Value'].quantile(0.9)

df['transformed_value'] = df['Value'].fillna(df['Value'].median())
df.loc[df['Value'] < P_low, 'transformed_value'] = P_low
df.loc[df['Value'] > P_high, 'transformed_value'] = P_high

print(df)

# hot deck imputation

data = {
    'Fruit': ['Apple', 'Banana', 'Orange', 'Grapes', np.nan, 'Apple', 'Banana', 'Orange', np.nan, 'Mango']
}

df = pd.DataFrame(data)
print(df.head())

missing_indices = df[df['Fruit'].isnull()].index

non_missing_values = df['Fruit'].dropna().values

np.random.seed(42)

imputed_values = np.random.choice(non_missing_values, size=len(missing_indices))

df.loc[missing_indices, 'Fruit'] = imputed_values

print(df)

# proxy variable imputation

data = {
    'Fruit': ['Apple', 'Banana', 'Orange', 'Grapes', np.nan, 'Apple', 'Banana', 'Orange', np.nan, 'Mango'],
    'Color': ['Red', 'Yellow', 'Orange', 'Green', 'Red', 'Red', 'Yellow', 'Orange', 'Green', 'Yellow']
}

df = pd.DataFrame(data)
print(df.head())

missing_indices = df[df['Fruit'].isnull()].index

fruit_mode_by_color = df.groupby('Color')['Fruit'].agg(lambda x: x.mode().iloc[0])

for idx in missing_indices:
    color = df.loc[idx, 'Color']
    imputed_fruit = fruit_mode_by_color[color]
    df.loc[idx, 'Fruit'] = imputed_fruit

print(df)

# multivariate imputation (MICE)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [3, np.nan, 5, np.nan, 7],
    'C': [np.nan, 2, 3, 4, np.nan],
    'D': [1, np.nan, 3, np.nan, 5]
}

df = pd.DataFrame(data)

print(df)

mice_imputer = IterativeImputer()

imputed_data = mice_imputer.fit_transform(df)

df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
print(df_imputed)

# KNN imputation

from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=2)

imputed_data = knn_imputer.fit_transform(df)

df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
print(df_imputed)