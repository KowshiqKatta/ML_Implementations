# handling date time features

import pandas as pd
import numpy as np

np.random.seed(0)
num_days = 100
date_rng = pd.date_range(start='2023-01-01', end='2023-04-10', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])

print(df.head())

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['day_of_week'] = df['date'].dt.dayofweek # extra feature

print(df.head())




