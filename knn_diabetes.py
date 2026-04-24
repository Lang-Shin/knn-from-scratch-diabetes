import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_missing_val(df, col):
    return (df[col] == 0).sum()


df = pd.read_csv("dataset/data.csv")

"""
    Handle missing data to columns such as Glucose, Blood Pressure, Skin Thickness, Insulin, BMI.
    Using median to fill missing data
"""

fix_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']      # Cols need to fix (0 -> median)

df[fix_cols] = df[fix_cols].replace(0, np.nan)                                  # replace 0 with NaN

for col in fix_cols:
    median = df[col].median()   
    df[col] = df[col].fillna(median)                                            # Replace NaN with median(col)

print(df.head(10), "\n\n")
print(count_missing_val(df, "Glucose"))
print(count_missing_val(df, "BloodPressure"))
print(count_missing_val(df, "SkinThickness"))
print(count_missing_val(df, "Insulin"))
print(count_missing_val(df, "BMI"))