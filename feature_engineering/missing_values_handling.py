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