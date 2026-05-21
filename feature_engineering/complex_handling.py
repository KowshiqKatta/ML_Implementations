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

# handling mixed features

data = {
    'Mixed_Feature': ['B123', 'C124', 'A120', 'B125', 'C126']
}

df = pd.DataFrame(data)

print(df.head())

df['Category'] = df['Mixed_Feature'].str[0] # extract the first character as category
df['Numerical'] = df['Mixed_Feature'].str[1:].astype(int) # extract the numeric part and convert to int

print(df.head())