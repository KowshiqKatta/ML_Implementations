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

# Under-sampling: Random Undersampling

class0_data = np.random.randn(900, 5)
class0_df = pd.DataFrame(class0_data, columns=[f'feature_{i}' for i in range(5)])
class0_df['class'] = 0

class1_data = np.random.randn(400, 5)
class1_df = pd.DataFrame(class1_data, columns=[f'feature_{i}' for i in range(5)])
class1_df['class'] = 1

df = pd.concat([class0_df, class1_df], ignore_index=True)

print(df.head())

print(df['class'].value_counts())

class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1]

print(class_0.shape)
print(class_1.shape)

undersampled_class_0 = class_0.sample(n=400, random_state=42)
print(undersampled_class_0.shape)

df_undersampled = pd.concat([undersampled_class_0, class_1], ignore_index=True)

print(df_undersampled['class'].value_counts())

df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_undersampled.head())