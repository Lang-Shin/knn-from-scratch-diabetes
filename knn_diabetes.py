import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_missing_val(df, col):
    return (df[col] == 0).sum()


df = pd.read_csv("dataset/data.csv")

print(df.head(10), "\n\n")
print(count_missing_val(df, "Glucose"))
print(count_missing_val(df, "BloodPressure"))
print(count_missing_val(df, "SkinThickness"))
print(count_missing_val(df, "Insulin"))
print(count_missing_val(df, "BMI"))