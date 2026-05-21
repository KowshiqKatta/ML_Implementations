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

# time series 

num_periods = 100
date_rng = pd.date_range(start='2023-01-01', periods=num_periods, freq='h')

df = pd.DataFrame(date_rng, columns=['timestamp'])
print(df.head())

df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second

print(df.head())

