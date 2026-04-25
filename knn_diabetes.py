import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


print(df)


X = df.drop("Outcome", axis=1)
y = df['Outcome']

mean = X.mean()
std = X.std()

X_scaled = (df - mean) / std

init_df = X_scaled.drop("Outcome", axis=1)

feat_scale_df = pd.concat([init_df, y], axis=1)

data_arr = feat_scale_df.values  # Convert to numpy array for faster math

feature = data_arr[:, :-1]      # features
labels = data_arr[:, -1]        # labeled data

