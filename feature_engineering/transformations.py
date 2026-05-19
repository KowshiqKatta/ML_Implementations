import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# log transformation

np.random.seed(0)

data = {
    'value': np.random.exponential(scale=2.0, size=1000)
}

df = pd.DataFrame(data)

print(df)

print(f"Skewness of 'value' column: {skew(df['value'])}")

df['log_transformed_value'] = np.log(df['value'] + 1) # Adding 1 to avoid log(0)

print(f"Skewness of 'log_transformed_value' column: {skew(df['log_transformed_value'])}")

# sqrt transformation

df['sqrt_transformed_value'] = np.sqrt(df['value'])

print(f"Skewness of 'sqrt_transformed_value' column: {skew(df['sqrt_transformed_value'])}")

