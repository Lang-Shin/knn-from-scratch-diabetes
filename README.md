GROUP 4:
BAJURA, CHARLES WILLIAM
CASTRO, CARLU

# knn-from-scratch-diabetes
A custom KNN machine learning classifier designed to predict diabetes diagnosis (0/1) based on patient health metrics. Includes preprocessing and manual distance-based classification.

**Course:** Computational Science for Computer Science
**Topic:** Machine Learning – K-Nearest Neighbors
**Dataset:** Pima Indians Diabetes Dataset (768 records)

---

## Project Structure

```
project/
├── dataset/
│   └── data.csv          # Diabetes dataset
├── main.py               # Main Python script
└── README.md
```


### 1. Data Preprocessing
- Loads `dataset/data.csv`
- Replaces zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with each column's **median** (since zero is biologically invalid for these features)
- Applies **standardization** (Z-score normalization) to scale all features

### 2. Train/Test Split
- Splits the dataset into **80% training / 20% testing** using a custom shuffle-based function (no sklearn)

### 3. KNN Implementation (From Scratch)
- Uses **Euclidean Distance** to find the K nearest neighbors
- Predicts diabetes outcome by **majority vote** among neighbors
- Tests K values: `1, 3, 5, 7, 9, 11, 13, 15`

### 4. Evaluation
- Computes **Accuracy** and **Confusion Matrix** for each K value

---

## Results

| K  | Accuracy |
|----|----------|
| 1  | 0.7078   |
| 3  | 0.7532   |
| 5  | 0.7338   |
| 7  | 0.7662   |
| 9  | 0.8117   |
| 11 | 0.7922   |
| 13 | 0.7987   |
| 15 | **0.8182** |

**Best K = 15** with an accuracy of **81.82%**

---

## BONUS: Logistic Regression Comparison

A manual Logistic Regression model was implemented from scratch using **Gradient Descent** (no sklearn) and trained on the same train/test split.

| Model                  | Accuracy |
|------------------------|----------|
| KNN (K=15)             | **0.8182** |
| Logistic Regression    | 0.7987   |

**KNN outperformed Logistic Regression** on this dataset and split.

---

## Output Graph

Running the script generates `bonus_results.png` containing two charts:

- **Left:** KNN Accuracy vs K Value — shows accuracy trend across all K values
- **Right:** Bar chart comparing KNN (best K) vs Logistic Regression accuracy

---

## Key Functions

| Function | Description |
|---|---|
| `custom_train_test()` | Shuffles and splits data into train/test sets |
| `predict_patient()` | Predicts diabetes for a single patient using KNN |
| `get_confusion_matrix()` | Returns TP, TN, FP, FN matrix |
| `calculate_accuracy()` | Computes prediction accuracy |
| `test_acc_cm()` | Runs evaluation across multiple K values |
| `sigmoid()` | Sigmoid activation for Logistic Regression |
| `logistic_train()` | Trains LR weights using gradient descent |
| `logistic_predict()` | Predicts outcomes using trained LR model |