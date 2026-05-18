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