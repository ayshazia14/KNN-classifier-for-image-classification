# Fashion MNIST Classification — K-Nearest Neighbours (KNN)

A supervised machine learning project that classifies Fashion MNIST clothing items using a K-Nearest Neighbours (KNN) classifier, with MinMax normalisation and k-fold cross-validation to find the optimal number of neighbours.

---

## Overview

This project applies KNN classification to the Fashion MNIST dataset — a benchmark image dataset of 28×28 grayscale images across 10 clothing categories. The pipeline includes data cleaning, normalisation, hyperparameter tuning via an elbow graph, and evaluation with a confusion matrix and classification report.

---

## Dataset

**Fashion MNIST** — loaded from local CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`)

Each row represents one image: a `label` column followed by 784 pixel values (28×28 flattened).

| Label | Category |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

> Download the dataset from [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and update the file paths in the notebook accordingly.

---

## Workflow

1. Load and inspect the training CSV
2. Check for null values and duplicates; drop duplicates
3. Subset to 500 samples for efficient cross-validation
4. Apply `MinMaxScaler` normalisation via a scikit-learn `Pipeline`
5. Run 5-fold cross-validation for k = 1 to 30 — track train and test error rates
6. Plot an elbow graph to identify the optimal k
7. Train final KNN model (k=5) on normalised training data
8. Evaluate with a confusion matrix heatmap and classification report
9. Transform and predict on the test set

---

## Results

| Metric | Value |
|---|---|
| Train Accuracy | Printed at runtime (%) |
| Optimal K | Identified via elbow graph |
| Evaluation | Confusion matrix + full classification report |

---

## Technologies Used

- Python
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn (KNeighborsClassifier, Pipeline, MinMaxScaler, cross_validate)

---

## How to Run

1. Download the Fashion MNIST CSVs from Kaggle and place them in your working directory.
2. Update the file paths in the notebook:
   ```python
   dataset = pd.read_csv("fashion-mnist_train.csv")
   dataset = pd.read_csv("fashion-mnist_test.csv")
   ```
3. Run the notebook:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   jupyter notebook Task_3.ipynb
   ```
