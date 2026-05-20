# Balancing Techniques: Random Oversampling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler

X = np.array([[1.0, 1.0], 
              [2.0, 2.0], 
              [5.0, 5.0], 
              [6.0, 6.0], 
              [7.0, 7.0],
              [8.0, 8.0], 
              [9.0, 9.0], 
              [10.0, 10.0], 
              [11.0, 11.0], 
              [12.0, 12.0]])

y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

df['class'] = y

print(df.head())

print(df['class'].value_counts())

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(df[['Feature1', 'Feature2']], df['class'])

df_resampled = pd.DataFrame(X_resampled, columns=['Feature1', 'Feature2'])

df_resampled['class'] = y_resampled

print(df_resampled['class'].value_counts())

# SMOTE (Synthetic Minority Over-sampling Technique)

from imblearn.over_sampling import SMOTE

np.random.seed(0)
num_samples = 1000
num_features = 5

X = np.random.randn(num_samples, num_features)
y = np.concatenate([np.zeros(900), np.ones(100)])

feature_names = [f'feature_{i}' for i in range(num_features)]

df = pd.DataFrame(X, columns=feature_names)
df['class'] = y.astype(int)

print(df.head())

print(df['class'].value_counts())

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(df.drop('class', axis=1), df['class'])

df_resampled = pd.DataFrame(X_resampled, columns=feature_names)
df_resampled['class'] = y_resampled

print(df_resampled['class'].value_counts())