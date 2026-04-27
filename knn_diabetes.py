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


#  Visualize Accuracy vs K + Compare with Logistic Regression

# Accuracy vs K

k_values = list(results.keys())
accuracies = [results[k]['accuracy'] for k in k_values]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, marker='o', color="#07852B")
plt.title('KNN Accuracy vs K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.ylim(0, 1)
plt.grid(True, alpha=0.6)


# --- Logistic Regression (Manual Implementation, No sklearn) ---

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_train(X, y, lr=0.1, epochs=1000):
    """Train Logistic Regression using Gradient Descent"""
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(epochs):
        z = X @ weights + bias
        pred = sigmoid(z)

        dw = (1 / m) * (X.T @ (pred - y))
        db = (1 / m) * np.sum(pred - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def logistic_predict(X, weights, bias, threshold=0.5):
    z = X @ weights + bias
    probs = sigmoid(z)
    return (probs >= threshold).astype(int)


# Train Logistic Regression on same train/test split
lr_weights, lr_bias = logistic_train(X_train, y_train)
lr_predictions = logistic_predict(X_test, lr_weights, lr_bias)

lr_accuracy = calculate_accuracy(y_test, lr_predictions)
lr_cm = get_confusion_matrix(y_test, lr_predictions)

print("\n\n========== Logistic Regression Comparison ==========")
print(f"Logistic Regression Accuracy : {lr_accuracy:.4f}")
print(f"Logistic Regression Confusion Matrix :\n{lr_cm}")

# Best KNN accuracy for comparison
best_k = max(results, key=lambda k: results[k]['accuracy'])
best_knn_acc = results[best_k]['accuracy']

print(f"\nBest KNN (K={best_k}) Accuracy   : {best_knn_acc:.4f}")
print(f"Logistic Regression Accuracy     : {lr_accuracy:.4f}")

# --- Bar Chart: KNN best vs Logistic Regression ---
plt.subplot(1, 2, 2)
models = [f'KNN (K={best_k})', 'Logistic Regression']
accs = [best_knn_acc, lr_accuracy]
colors = ['steelblue', 'tomato']

bars = plt.bar(models, accs, color=colors, width=0.4)
plt.title('KNN vs Logistic Regression')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.6)

plt.tight_layout()
plt.show()