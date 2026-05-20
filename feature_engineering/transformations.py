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

# reciprocal transformation

data = np.random.exponential(scale=2.0, size=1000)

data = 50 - data

df = pd.DataFrame(data, columns = ['value'])

print(skew(df['value']))

df['reciprocal_transformed_value'] = 1 / (df['value'] + 1) # Adding 1 to avoid division by zero

print(f"Skewness of 'reciprocal_transformed_value' column: {skew(df['reciprocal_transformed_value'])}")


# Box-Cox transformation

np.random.seed(0)

data = {
    'A': np.random.exponential(scale=2.0, size=1000),
    'B': np.random.chisquare(df=2, size=1000)
}

df = pd.DataFrame(data)

print(df)

print(f"Skewness of 'A' column: {skew(df['A'])}")   
print(f"Skewness of 'B' column: {skew(df['B'])}")

from scipy.stats import boxcox

df["Transformed_A"], lambda_value_a = boxcox(df["A"]) 

print(f"Lambda for 'A': {lambda_value_a}")
print(f"Skewness of 'Transformed_A' column: {skew(df['Transformed_A'])}")

df["Transformed_B"], lambda_value_b = boxcox(df["B"])

print(f"Lambda for 'B': {lambda_value_b}")
print(f"Skewness of 'Transformed_B' column: {skew(df['Transformed_B'])}")






