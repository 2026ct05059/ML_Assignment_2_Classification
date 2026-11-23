# Machine Learning Classification Model Comparison

## 1. Problem Statement

The objective of this project is to implement multiple machine learning classification models using one classification dataset and compare their performance using standard evaluation metrics.

The project implements the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model is evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 2. Dataset Description

### Dataset Name

**UCI Adult / Census Income Dataset**

### Dataset Source

UCI Machine Learning Repository

### Dataset URL

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

### Problem Type

Binary classification.

### Target Variable

The target variable is:

`income`

The target is converted into binary values for model training:

- `<=50K` → `0`
- `>50K` → `1`

### Dataset Size

The original dataset loaded in the notebook contains:

- **32,561 instances**
- **14 input features**
- **1 target column**

After duplicate removal:

- **32,537 instances**
- **14 input features**

### Feature Types

#### Numerical Features

- age
- fnlwgt
- education_num
- capital_gain
- capital_loss
- hours_per_week

#### Categorical Features

- workclass
- education
- marital_status
- occupation
- relationship
- race
- sex
- native_country

### Missing Values

The notebook identified the following missing values before preprocessing:

| Feature | Missing Values | Percentage |
|---|---:|---:|
| occupation | 1,843 | 5.6601% |
| workclass | 1,836 | 5.6386% |
| native_country | 583 | 1.7905% |
| Other features | 0 | 0% |

### Duplicate Records

The notebook identified:

- **24 duplicate rows**

After removing duplicates:

- **32,537 rows** remained.

### Class Distribution

The target class distribution after duplicate removal was:

| Income Class | Count | Percentage |
|---|---:|---:|
| `<=50K` | 24,698 | 75.9074% |
| `>50K` | 7,839 | 24.0926% |

The dataset therefore contains a noticeable class imbalance, which was examined before model training.

---

## 3. GitHub Repository Link

**Repository:** https://github.com/2026ct05059/ML_Assignment_2_Classification

---

## 4. Data Preprocessing

The preprocessing implemented in the Colab notebook consists of the following steps.

### 4.1 Missing Value Handling

Missing values are handled inside the preprocessing pipeline.

For numerical features:

- `SimpleImputer(strategy="median")`

For categorical features:

- `SimpleImputer(strategy="most_frequent")`

### 4.2 Duplicate Handling

Duplicate rows are identified and removed before model training.

The dataset contained 24 duplicate rows, which were removed.

### 4.3 Categorical Encoding

Categorical features are converted into numerical form using:

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 4.4 Feature Scaling

Numerical features are scaled using:

`StandardScaler()`

### 4.5 Train-Test Split

The notebook uses an 80:20 train-test split with stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

The resulting dataset sizes were:

- **Training rows:** 26,029
- **Testing rows:** 6,508

### 4.6 Fixed Random Seed

A fixed random seed is used:

`random_state = 42`

### 4.7 Data Leakage Prevention

The preprocessing pipeline is fitted on the training data and then applied to the test data.

This ensures that information from the test set is not used during preprocessing or model training.

---

## 5. Models Used

### 5.1 Logistic Regression

Logistic Regression is used as a baseline binary classification model.

Configuration:

```text
max_iter = 1000
random_state = 42
```

Source file:

`model/logistic_regression.py`

---

### 5.2 Decision Tree Classifier

Decision Tree Classifier predicts the target class using feature-based decision rules.

Configuration:

```text
random_state = 42
```

Source file:

`model/decision_tree.py`

---

### 5.3 K-Nearest Neighbor Classifier

KNN predicts the class of an observation based on its nearest neighboring observations.

Configuration:

```text
n_neighbors = 5
```

Source file:

`model/knn.py`

---

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes is used as the Naive Bayes implementation.

The model is trained after applying the preprocessing transformation to the training data.

Source file:

`model/naive_bayes.py`

---

### 5.5 Random Forest Classifier

Random Forest is the ensemble model used in this project.

Configuration:

```text
n_estimators = 200
random_state = 42
n_jobs = -1
```

Source file:

`model/random_forest.py`

---

## 6. Evaluation Metrics

The following metrics are calculated for each model.

### Accuracy

Measures the proportion of correctly classified observations.

### AUC Score

Measures the ability of the model to distinguish between the two classes.

### Precision

Measures how many of the observations predicted as positive are actually positive.

### Recall

Measures how many of the actual positive observations are correctly identified.

### F1 Score

The harmonic mean of Precision and Recall.

### Matthews Correlation Coefficient (MCC)

MCC provides a balanced classification-performance measure based on all four confusion-matrix categories.

---

## 7. Model Comparison

All five models were trained and evaluated on the same dataset using the same 80:20 stratified train-test split and the same preprocessing approach.

The final results obtained from the Colab notebook are:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8569 | 0.9099 | 0.7437 | 0.6199 | 0.6762 | 0.5893 |
| Decision Tree | 0.8207 | 0.7539 | 0.6286 | 0.6250 | 0.6268 | 0.5088 |
| KNN | 0.8379 | 0.8653 | 0.6778 | 0.6237 | 0.6496 | 0.5452 |
| Naive Bayes | 0.5361 | 0.7492 | 0.3369 | 0.9560 | 0.4983 | 0.3300 |
| **Random Forest (Ensemble)** | **0.8619** | **0.9092** | **0.7451** | **0.6486** | **0.6935** | **0.6073** |

### Average Score Comparison

The notebook also calculated the average of the six evaluation metrics for each model:

| ML Model Name | Average Score |
|---|---:|
| **Random Forest** | **0.744254** |
| Logistic Regression | 0.732657 |
| KNN | 0.699907 |
| Decision Tree | 0.660630 |
| Naive Bayes | 0.567749 |

Based on the average score across the six metrics, **Random Forest is the overall winner** for this dataset.

### Key Comparison

- Random Forest achieved the highest Accuracy.
- Logistic Regression achieved the highest AUC.
- Random Forest achieved the highest Precision.
- Naive Bayes achieved the highest Recall.
- Random Forest achieved the highest F1 Score.
- Random Forest achieved the highest MCC.
- Random Forest achieved the highest average score across the six metrics.

---

## 8. Visualization

The notebook generated two visual representations of model performance.

### 8.1 Classification Model Performance Comparison

A heatmap was created using:

- Accuracy
- AUC
- Precision
- Recall
- F1
- MCC

The heatmap is titled:

`Classification Model Performance Comparison`

### 8.2 Model Performance Comparison Chart

A bar chart was generated to compare the six evaluation metrics across all five models.

The chart uses:

- ML Model Name on the X-axis
- Metric Score on the Y-axis

These visualizations should be included in the final project / README as required.

---

## 9. Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved an Accuracy of **0.8569** and the highest AUC of **0.9099**. It also achieved Precision of **0.7437**, Recall of **0.6199**, F1 Score of **0.6762**, and MCC of **0.5893**. It provided strong and relatively balanced baseline performance and ranked second by average score. |
| **Decision Tree** | Decision Tree achieved an Accuracy of **0.8207**, AUC of **0.7539**, Precision of **0.6286**, Recall of **0.6250**, F1 Score of **0.6268**, and MCC of **0.5088**. Its performance was below Logistic Regression, KNN and Random Forest across most metrics. |
| **KNN** | KNN achieved an Accuracy of **0.8379**, AUC of **0.8653**, Precision of **0.6778**, Recall of **0.6237**, F1 Score of **0.6496**, and MCC of **0.5452**. It performed better overall than the Decision Tree, but below Logistic Regression and Random Forest. |
| **Naive Bayes** | Naive Bayes achieved an Accuracy of **0.5361**, AUC of **0.7492**, Precision of **0.3369**, Recall of **0.9560**, F1 Score of **0.4983**, and MCC of **0.3300**. It achieved the highest Recall but had very low Precision, F1 and MCC, resulting in the weakest overall performance. |
| **Random Forest (Ensemble)** | Random Forest achieved the highest Accuracy (**0.8619**), Precision (**0.7451**), Recall (**0.6486**), F1 Score (**0.6935**) and MCC (**0.6073**). Its AUC of **0.9092** was also very close to Logistic Regression's highest AUC of **0.9099**. It achieved the highest average score of **0.744254** and was selected as the overall winner. |

### Overall Winner

**Random Forest**

Random Forest was selected as the overall winner because it achieved the highest average score across Accuracy, AUC, Precision, Recall, F1 Score and MCC.

It also achieved the highest Accuracy, Precision, Recall, F1 Score and MCC among the five evaluated models.

---

## 10. Project Structure

```text
ML_Assignment_2_Classification/
│
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── ml_pipeline_utils.py
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    └── random_forest.py
```

---

## 11. Requirements

The project uses the following Python libraries:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## 12. Execution Instructions

### Google Colab

The project was developed and executed in Google Colab.

Execution steps:

1. Import the required Python libraries.
2. Load the UCI Adult dataset.
3. Inspect the dataset.
4. Identify missing values.
5. Identify and remove duplicate records.
6. Examine class distribution.
7. Separate input features and target variable.
8. Identify numerical and categorical features.
9. Perform the 80:20 stratified train-test split.
10. Apply preprocessing using the training data.
11. Train all five classification models.
12. Calculate Accuracy, AUC, Precision, Recall, F1 and MCC.
13. Generate the model comparison table.
14. Generate the heatmap and bar chart.
15. Export `test_data.csv` and `metrics.csv`.

### Running from GitHub

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Open the project:

```bash
cd ML_Assignment_2_Classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the individual model files:

```bash
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py
```

---

## 13. Test Data

The test data used in the experiment is stored in:

`test_data.csv`

The notebook created this file from the 20% test split used during model evaluation.

The test set contains:

- **6,508 records**

and includes the input features together with the binary target column:

`income_binary`

---

## 14. Reproducibility

The project uses:

```text
random_state = 42
```

The train-test split is stratified, and the same dataset and preprocessing approach are used consistently across all five classification models.

---

## 15. BITS Virtual Lab Execution

The completed project must also be executed in the **BITS Virtual Lab**.

A screenshot of the assignment execution in the BITS Virtual Lab should be captured and included in the final submission PDF.

---

## 16. Conclusion

This project implemented and compared five machine learning classification algorithms using the UCI Adult / Census Income dataset.

The preprocessing included:

- Missing-value handling
- Duplicate removal
- Categorical encoding
- Numerical feature scaling
- 80:20 stratified train-test split
- Fixed random seed
- Training-only preprocessing

The five models were evaluated using six metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

Based on the final experimental results, **Random Forest achieved the highest overall average score of 0.744254 and was selected as the overall winner**.

---

## 17. Academic Integrity

This project should be reviewed, understood, tested, and adapted before submission.

All reported metrics, observations, visualizations and conclusions should correspond to the actual execution of the project.
