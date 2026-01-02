
# Wine Analysis, Classification, and Quality Prediction Project


## 1. Introduction

This project focuses on exploring and modeling a wine dataset containing both physicochemical measurements and human quality ratings.
Two main predictive tasks are considered:

1. **Classifying wine type** (red vs white).
2. **Predicting wine quality** using regression.

Beyond predictive modeling, the goal is to perform a complete end-to-end analysis: understanding the data, cleaning it, visualizing patterns, building models, interpreting results, and identifying potential improvements.

## Chosen dataset 

For this project, the **Wine Type Classification Dataset** was selected. This dataset combines two well-known wine datasets originally published in the UCI Machine Learning Repository: one for red wines and one for white wines. The combined version contains **6,497 samples** and **13 physicochemical attributes**, along with the wine type and quality score.


### Variables Description

The table below lists all variables included in the dataset, along with their meaning and measurement units when available.

| Feature              | Description                                                | Unit                    |
| -------------------- | ---------------------------------------------------------- | ----------------------- |
| fixed acidity        | Non-volatile acids primarily responsible for total acidity | g/dm³ (tartaric acid)   |
| volatile acidity     | Acetic acid content associated with vinegar taste          | g/dm³ (acetic acid)     |
| citric acid          | Citric acid concentration contributing to freshness        | g/dm³                   |
| residual sugar       | Remaining sugar after fermentation                         | g/dm³                   |
| chlorides            | Salt content in wine                                       | g/dm³ (sodium chloride) |
| free sulfur dioxide  | Free form of SO₂ preventing microbial growth               | mg/dm³                  |
| total sulfur dioxide | Total concentration of SO₂ (bound + free)                  | mg/dm³                  |
| density              | Density of wine relative to water                          | g/cm³                   |
| pH                   | Acidity level (lower = more acidic)                        | — (dimensionless)       |
| sulphates            | Potassium sulphate used as preservative                    | g/dm³                   |
| alcohol              | Alcohol percentage by volume                               | % vol                   |
| quality              | Sensory quality score typically rated 0–10                 | score                   |
| type                 | Wine type (Red or White)                                   | categorical             |


## 2. Objectives

The project has three core objectives.

### 2.1 Dataset Understanding and Exploration

* Perform a detailed exploratory analysis.
* Identify trends, correlations, anomalies, and potential data quality issues.
* Understand how physicochemical properties vary between wines and how they relate to quality.

### 2.2 Classification: Predicting Wine Type

* Build supervised learning models that classify each sample as **red** or **white**.
* Compare different algorithms.
* Evaluate performance rigorously and interpret results.

### 2.3 Regression: Predicting Wine Quality

* Develop regression models to estimate numerical quality scores.
* Identify which features most strongly influence predictions.
* Evaluate predictive accuracy and highlight limitations.

---

## 3. Dataset Description

The dataset includes:

* A categorical label:

  * `type` (red or white)

* A numerical target for regression:

  * `quality` (integer rating)

* Several physicochemical features, such as:

  * fixed acidity
  * volatile acidity
  * citric acid
  * residual sugar
  * chlorides
  * free sulfur dioxide
  * total sulfur dioxide
  * sulphates
  * alcohol
  * pH

These variables describe measurable chemical characteristics that influence taste, preservation, and sensory evaluation.

---

## 4. Project Workflow Overview

The project follows a structured data science lifecycle:

1. Problem definition
2. Data loading and inspection
3. Data cleaning and preprocessing
4. Exploratory data analysis (EDA)
5. Feature engineering and transformation
6. Model building for classification and regression
7. Model evaluation and tuning
8. Interpretation of results
9. Reporting and conclusions

Each of these steps is detailed below.

---

## 5. Data Preparation

### 5.1 Data Loading

* Load the dataset from CSV.
* Inspect structure: number of rows, columns, and data types.

### 5.2 Data Quality Checks

* Detect missing values.
* Identify duplicates.
* Look for inconsistent values.

Possible actions:

* Remove duplicate rows. OK
* Handle missing values using deletion or imputation. OK
* Verify ranges and units for variables. OK

### 5.3 Encoding Categorical Variables

* Convert `type` into numerical format when needed for modeling.

### 5.4 Feature Scaling

Some models require normalization due to variables having different scales.

Consider:

* Standardization (mean = 0, std = 1)
* Min-Max scaling (values normalized between 0 and 1)

Scaling should be applied inside pipelines to avoid data leakage.

### 5.5 Train/Test Split

Split the dataset into training and testing subsets to evaluate generalization.

---

## 6. Exploratory Data Analysis (EDA)

### 6.1 Descriptive Statistics

* Mean, median, minimum, maximum, standard deviation.
* Compare distributions across wine types.

### 6.2 Visual Exploration

Useful visualizations include:

* Histograms
* Boxplots for outlier detection
* Scatter plots of key feature relationships
* Distribution comparisons between red and white wine

### 6.3 Correlation Analysis

* Compute correlation matrix.
* Investigate which features are associated with:

  * wine quality
  * type classification

Important notes:
Correlation indicates association, not causation.

---

## 7. Classification: Predicting Wine Type

### 7.1 Problem Definition

Binary classification:

* Input: physicochemical features
* Output: wine type (red/white)

### 7.2 Candidate Models

Consider evaluating multiple models:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting (e.g., XGBoost, LightGBM)
* Support Vector Machine
* Simple Neural Network

### 7.3 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Pay attention to class imbalance if one wine type is more frequent.

---

## 8. Regression: Predicting Wine Quality

### 8.1 Problem Definition

Numeric prediction:

* Input: chemical features
* Output: quality score

### 8.2 Candidate Models

* Linear Regression
* Ridge / Lasso Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost / LightGBM Regressor
* Multi-Layer Perceptron Regressor

### 8.3 Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² score

Models should be compared using cross-validation.

---

## 9. Model Optimization

### 9.1 Hyperparameter Tuning

Use techniques such as:

* Grid Search
* Random Search
* Bayesian Optimization (optional)

### 9.2 Cross-Validation

Apply k-fold cross-validation to ensure reliable evaluation.

### 9.3 Avoiding Overfitting

* Regularization
* Early stopping (for boosting / neural networks)
* Proper data splitting
* Feature selection if needed

---

## 10. Model Interpretation

Understanding why predictions are made is essential.

Possible techniques:

* Feature importance for tree-based models
* Coefficients for linear models
* SHAP or LIME for local interpretability

Questions to investigate:

* Which variables best distinguish red vs white wine?
* Which factors increase predicted quality the most?
* Are there surprising or counter-intuitive relationships?

---

## 11. Validation and Robustness Checks

* Confirm that preprocessing pipelines are consistent.
* Verify absence of data leakage.
* Test models on unseen data.
* Assess stability across different random splits.

---

## 12. Documentation and Reporting

A complete final report should include:

* Project goals
* Dataset description
* Methodology
* EDA results
* Model comparison
* Interpretation of findings
* Discussion of limitations
* Future improvement ideas

---

## 13. Possible Extensions

Potential directions to enhance the project:

* Add new derived features (ratios, interactions, transformations)
* Try multi-task learning (predict type and quality jointly)
* Analyze sensory study bias in quality ratings
* Investigate domain knowledge links to winemaking processes

---

## 14. Conclusion

This project integrates exploratory analysis, supervised classification, and regression modeling within a single dataset. By following the structured workflow outlined above, it becomes possible to:

* understand the dataset deeply,
* develop robust predictive models,
* extract meaningful insights about factors influencing wine type and quality.

The end goal is not only predictive performance but also clear interpretation and scientific reasoning behind the models.

---

If you want, I can now:

* adapt this document to your specific dataset version,
* help you transform it into a formal report,
* or build the full notebook structure step by step.
