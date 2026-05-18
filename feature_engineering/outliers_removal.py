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