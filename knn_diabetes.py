import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def custom_train_test(features, labels, train_size=0.8):
    num_samples = len(features)

    indices = np.arange(num_samples)

    np.random.shuffle(indices)  # shuffle data to avoid bias

    split_point = (train_size * num_samples)

    train_data_id = indices[ : int(split_point)]  
    test_data_id = indices[int(split_point) : ]

    X_train, X_test = features[train_data_id], features[test_data_id]
    y_train, y_test = labels[train_data_id], labels[test_data_id]

    return X_train, X_test, y_train, y_test


def get_distances(test_point, X_train):
    difference = test_point - X_train

    distances = np.sqrt(np.sum(difference**2), axis=1)

    return distances


df = pd.read_csv("dataset/data.csv")

"""
    Handle missing data to columns such as Glucose, Blood Pressure, Skin Thickness, Insulin, BMI.
    Using median to fill missing data
"""
fix_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']      # Cols need to fix (0 -> median)

df[fix_cols] = df[fix_cols].replace(0, np.nan)      # replace 0 with NaN

for col in fix_cols:
    median = df[col].median()   
    df[col] = df[col].fillna(median)                # Replace NaN with median(col)


X = df.drop("Outcome", axis=1)
y = df['Outcome']

mean = X.mean()
std = X.std()

X_scaled = (df - mean) / std

init_df = X_scaled.drop("Outcome", axis=1)          # Remove the outcome col in the feature scalled data

feat_scale_df = pd.concat([init_df, y], axis=1)     # Join feature scalled data with labeled one

data_arr = feat_scale_df.values  # Convert to numpy array for faster math

features = data_arr[:, :-1]      # features
labels = data_arr[:, -1]        # labeled data

X_train, X_test, y_train, y_test = custom_train_test(features, labels)

patient = X_test[0]

