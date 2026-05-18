# standardization

import numpy as np
import pandas as pd

ages = np.random.normal(loc = 40, scale = 10, size = 1000)
salaries = np.random.normal(loc = 50000, scale = 15000, size = 1000)

data_df = pd.DataFrame({'Age': ages, 'Salary': salaries})

print(data_df.head())

import matplotlib.pyplot as plt

plt.hist(data_df['Age'])
plt.hist(data_df['Salary'])
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

standardized_data = scaler.fit_transform(data_df)
standardized_df = pd.DataFrame(standardized_data, columns=data_df.columns)

print(standardized_data)
print(standardized_df.head())

# normalization

data = np.random.uniform(low = 10, high = 100, size = (1000, 2))
data_df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

print(data_df.head())

plt.hist(data_df['Feature1'])
plt.hist(data_df['Feature2'])

plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_df) 
scaled_df = pd.DataFrame(scaled_data, columns=data_df.columns)

print(scaled_data)
print(scaled_df.head())