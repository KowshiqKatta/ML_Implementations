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
