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




