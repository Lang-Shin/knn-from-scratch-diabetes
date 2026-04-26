import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def custom_train_test(features, labels, train_size=0.8):
    """Train features and labels"""

    num_samples = len(features)

    indices = np.arange(num_samples)

    np.random.shuffle(indices)  # shuffle data to avoid bias

    split_point = (train_size * num_samples)            # 80% of the dataset to be train

    train_data_id = indices[ : int(split_point)]  
    test_data_id = indices[int(split_point) : ]

    X_train, X_test = features[train_data_id], features[test_data_id]       # train and test features
    y_train, y_test = labels[train_data_id], labels[test_data_id]           # train and test labels

    return X_train, X_test, y_train, y_test


def predict_patient(test_point, X_train, y_train, k):
    """Predicts if the test_point(patient) has diabetes according to its nearest neighbors"""

    difference = test_point - X_train                           # (x2 - x1) (y2 - y1)

    distances = np.sqrt(np.sum(difference**2, axis=1))

    nearest_neighbors_indices = np.argsort(distances)[:k]       # sort indices

    neighbor_labels = y_train[nearest_neighbors_indices]

    prediction = 1 if np.sum(neighbor_labels) > (k/2) else 0

    return prediction

def get_confusion_matrix(y_test, predictions):

    TP = np.sum((y_test == 1) & (predictions == 1))
    TN = np.sum((y_test == 0) & (predictions == 0))
    FP = np.sum((y_test == 0) & (predictions == 1))
    FN = np.sum((y_test == 1) & (predictions == 0))

    return np.array([[TP, TN], [FP, FN]])


def calculate_accuracy(y_test, predictions):
    """
                                num of correct predictions
    Calculate Accuracy  = ------------------------------------
                                    total predictions
    """
    correct = np.sum(y_test == predictions)

    return correct/len(y_test)


def test_acc_cm(n, X_test, y_test, X_train, y_train):
    """Test model accuracy"""
    results = {}

    for k in n:
        predictions = np.array([predict_patient(tp, X_train, y_train, k) for tp in X_test])

        acc = calculate_accuracy(y_test, predictions)
        cm = get_confusion_matrix(y_test, predictions)

        results[k] = {
            "accuracy" : acc,
            "confusion_matrix" : cm
        }

    return results

def smanualh_icomputej(patient, samples, labels, k):
    """Manual Computing Essentials :)"""

    print("\nPatient : ", patient)

    print("\n10 Training samples : ")
    for i in range(10):
        print(samples[i])

    print("\n10 Trainning labels : ")
    for i in range(10):
        print(labels[i])


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

"""
Manual Solving
"""
# patient = X_test[0]
# ten_training_samples = X_train[0:10]        
# ten_training_labels = y_train[0:10]

# print("\n\n", feat_scale_df.head(20))

# smanualh_icomputej(patient, ten_training_labels, ten_training_samples, 5)

# print("Prediction : ", predict_patient(patient, X_train, y_train, 7))

n = 8
k = np.arange(1, 2*n+1, 2)

results = test_acc_cm(k, X_test, y_test, X_train, y_train)

for i in range(n):
    print(f"K = {k[i]}")
    print(f"Accuracy : {results[k[i]]['accuracy']:.4f}")
    print(f"Confusion Matrix : {results[k[i]]['confusion_matrix']}\n\n")