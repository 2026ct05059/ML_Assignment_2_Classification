# Machine Learning Classification Model Comparison

## 1. Problem Statement

The objective of this project is to develop and compare machine learning
classification models to predict whether an individual's annual income
is `<=50K` or `>50K`.

The target variable is `income`:

-   `0` -- Income `<=50K`
-   `1` -- Income `>50K`

Five classification algorithms are implemented and evaluated to identify
the best-performing model.

## 2. Dataset Description

**Dataset:** UCI Adult / Census Income Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:**
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

-   Initial records: 32,561
-   Input features: 14
-   Target variable: `income`
-   Data types: Numerical and categorical

### Data Cleaning

-   Identified missing values in the dataset.
-   Identified and removed 24 duplicate records.
-   Handled missing numerical values using median imputation.
-   Handled missing categorical values using most-frequent imputation.
-   Applied standard scaling to numerical features.
-   Applied one-hot encoding to categorical features.

After duplicate removal, the dataset contained **32,537 records**.

### Train-Test Split

The cleaned dataset was divided into:

-   Training data: 80% (26,029 records)
-   Testing data: 20% (6,508 records)

Stratified sampling was used to maintain the class distribution. A fixed
random state of `42` was used for reproducibility.

Preprocessing transformations were fitted using only the training data.

## 3. GitHub Repository

**GitHub Repository:** ML Assignment 2 -- Classification

**Repository Link:**
https://github.com/2026ct05059/ML_Assignment_2_Classification

## 4. Machine Learning Algorithms

The following five classification algorithms were implemented:

1.  Logistic Regression
2.  Decision Tree
3.  K-Nearest Neighbors (KNN)
4.  Gaussian Naive Bayes
5.  Random Forest

### Model Performance Comparison

The models were evaluated using Accuracy, AUC, Precision, Recall, F1
Score, and Matthews Correlation Coefficient (MCC).

  -------------------------------------------------------------------------------
  Model          Accuracy        AUC   Precision     Recall   F1 Score        MCC
  ------------ ---------- ---------- ----------- ---------- ---------- ----------
  Logistic         0.8569     0.9099      0.7437     0.6199     0.6762     0.5893
  Regression                                                           

  Decision         0.8207     0.7539      0.6286     0.6250     0.6268     0.5088
  Tree                                                                 

  KNN              0.8379     0.8653      0.6778     0.6237     0.6496     0.5452

  Naive Bayes      0.5361     0.7492      0.3369     0.9560     0.4983     0.3300

  Random           0.8619     0.9092      0.7451     0.6486     0.6935     0.6073
  Forest                                                               
  -------------------------------------------------------------------------------

## 5. Model Observations

  -----------------------------------------------------------------------
  ML Model Name                       Observation about Model Performance
  ----------------------------------- -----------------------------------
  Logistic Regression                 Achieved 85.69% accuracy and the
                                      highest AUC of 0.9099. It provided
                                      strong baseline performance and
                                      ranked second by average score.

  Decision Tree                       Achieved 82.07% accuracy and 0.7539
                                      AUC. Its performance was lower than
                                      Logistic Regression, KNN and Random
                                      Forest across most metrics.

  K-Nearest Neighbors (KNN)           Achieved 83.79% accuracy and 0.8653
                                      AUC. It demonstrated reasonable
                                      classification performance and
                                      ranked third by average score.

  Gaussian Naive Bayes                Achieved the highest recall of
                                      95.60%. However, its precision of
                                      33.69% and accuracy of 53.61%
                                      indicate a high number of
                                      false-positive predictions.

  Random Forest                       Achieved the highest accuracy,
                                      precision, recall, F1 Score and MCC
                                      among all five models. It
                                      demonstrated the best overall
                                      classification performance based on
                                      the average of the six metrics.
  -----------------------------------------------------------------------

## 6. Visualization

The performance of all five classification algorithms was compared
through graphical representations of:

-   Accuracy
-   AUC
-   Precision
-   Recall
-   F1 Score
-   MCC

The notebook includes:

-   A heatmap titled **Classification Model Performance Comparison**.
-   A grouped bar chart titled **Model Performance Comparison**.

These visualizations compare the six evaluation metrics across all five
classification models.

## 7. Overall Winner

**Random Forest** was selected as the overall best-performing model.

It achieved:

-   Accuracy: 86.19%
-   AUC: 0.9092
-   Precision: 74.51%
-   Recall: 64.86%
-   F1 Score: 0.6935
-   MCC: 0.6073

Random Forest achieved the highest average score of **0.744254** across
the six evaluation metrics. It achieved the best result in five out of
the six individual metrics. Logistic Regression achieved the highest AUC
of 0.9099.

Therefore, Random Forest was selected as the overall winner among the
five evaluated classification algorithms.

## 8. Repository Contents

-   `ML_Assignment_2_Classification.ipynb` -- Complete implementation,
    data analysis, model training and evaluation.
-   `test_data.csv` -- Test dataset used for model evaluation.
-   `requirements.txt` -- Required Python libraries.
-   `model/` -- Individual classification model scripts and shared
    pipeline utilities:
    -   `ml_pipeline_utils.py`
    -   `logistic_regression.py`
    -   `decision_tree.py`
    -   `knn.py`
    -   `naive_bayes.py`
    -   `random_forest.py`
