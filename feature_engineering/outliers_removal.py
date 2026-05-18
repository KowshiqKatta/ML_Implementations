# z score method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

data = {
    'value': [10, 12, 12, 13, 12, 11, 14, 13, 100, 12, 11, 13, 12, 14, 13, 12, 11, 13, 12, 14]
}

df = pd.DataFrame(data)
print(df)

df['z_score'] = zscore(df['value'])
print(df)

threshold = 3

df['outlier'] = np.abs(df['z_score']) > threshold
print(df)

df_clean = df[df['outlier'] == False]
print(df_clean)

# IQR method

data = {
    'Value': np.append(np.random.normal(50, 10, 100), [150, 200, 250, 300, 350, 400])
}

df = pd.DataFrame(data)
print(df)

Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['outlier'] = (df['Value'] < lower_bound) | (df['Value'] > upper_bound)
print(df)   

df_clean = df[df['outlier'] == False]
print(df_clean)

# percentile method

data = np.concatenate([np.random.normal(0, 1, 90), np.array([10, 12, 15, -8, -10])])

df = pd.DataFrame(data, columns=['Value'])

print(df)

lower_bound = np.percentile(df['Value'], 5)
upper_bound = np.percentile(df['Value'], 95)

lower_bound1 = df['Value'].quantile(0.05)
upper_bound1 = df['Value'].quantile(0.95)

print(lower_bound, upper_bound)
print(lower_bound1, upper_bound1)

df['outlier'] = (df['Value'] < lower_bound) | (df['Value'] > upper_bound)
print(df)

df_clean = df[df['outlier'] == False]
print(df_clean)

# winsorization method

data = np.concatenate([np.random.normal(0, 1, 50), np.array([10, 12, 15, -8, -10])])

df = pd.DataFrame(data, columns=['Value'])

print(df)

lower_bound = np.percentile(df['Value'], 5)
upper_bound = np.percentile(df['Value'], 95)

print(lower_bound, upper_bound)

df['winsorized'] = df['Value'].clip(lower_bound, upper_bound)

print(df)