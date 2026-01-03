# ITML_final_project - Wine type and quality analysis

- **Date**: 2026-01-03
- **Status**: In progress
- **Author**: Mathieu Adnet, Colin MANYRI
- **License**: MIT - Copyright (c) 2026 Colin MANYRI & Mathieu Adnet
- **Version**: 10.0.0.1

This repository contains the final project developed as part of the university course “Introduction to Machine Learning” at Politechnika Krakowska. The goal of the project is to apply fundamental ML concepts to a complete workflow — from data preprocessing to model evaluation.

## Project Goal

The goal of this project is to apply and deepen the concepts introduced in the *Introduction to Machine Learning* course by carrying out a complete end-to-end analysis of the **Wine Type Classification Dataset**. **Kaggle access: [Wine Type Classification Dataset](https://www.kaggle.com/datasets/ehsanesmaeili/red-and-white-wine-quality-merged)**

This dataset contains **6,497 wine samples** (red and white) and provides **13 numerical features** describing chemical properties and sensory quality. It is suitable for multiple machine learning tasks such as classification, regression, and exploratory analysis.

The main objectives of this project are:

### Dataset Understanding & Preprocessing

* Perform exploratory data analysis (EDA)
* Clean and prepare the dataset
* Apply normalization and feature preprocessing techniques
* Validate dataset splits to avoid leakage and bias

### Machine Learning Modeling

We implement and compare multiple machine learning methods covered during the course, including:

* Models for **binary classification**
  → predicting whether a wine is **Red** or **White**

* Models for **regression**
  → predicting the wine **quality score** (0–10 scale)

Performance comparison focuses on accuracy, generalization ability, and robustness.

### Model Evaluation & Interpretation

* Compare different models on consistent metrics
* Analyze strengths, weaknesses, and trade-offs
* Discuss overfitting, underfitting, and model complexity
* Interpret relationships between chemical properties and outcomes

### Automated Reporting

The project also includes an **auto-generated reporting system** that produces detailed analysis summaries to help users understand results and support decision-making.


This work is carried out as part of the academic program at **Politechnika Krakowska**, and culminates in both a written report and an oral presentation.


## Repository Structrure

### `data/`

This directory contains all datasets used throughout the project lifecycle.
It includes:

* raw source tables,
* cleaned and formatted versions ready for modeling,
* intermediate datasets created after transformations such as normalization, feature selection, or PCA.

These files support both exploratory analysis and model development without needing to repeatedly regenerate processed data.

---

### `documents/`

Repository for written project material and supporting documentation.
It contains planning notes, descriptions of experiments, and deliverables used for reporting progress.

#### `documents/reports/`

Dedicated space for generated analytical outputs.
It contains performance summaries, experiment logs, model comparison summaries, and structured reports intended for interpretation and presentation.

---

### `plots/`

Folder containing all visual outputs (figures, comparison charts, diagnostic plots).
These visuals are produced during exploratory data analysis, modeling stages, and final reporting.

---

### `src/`

Core working notebooks used to perform analysis, build models, and create deliverables.

#### [classification.ipynb](src/classification.ipynb)

Notebook dedicated to training and evaluating machine learning models for **binary wine type classification (Red vs White)**.
Covers preprocessing integration, model construction, training procedure, metrics evaluation, and visual interpretation of classification performance.

#### [data_explore.ipynb](src/data_explore.ipynb)

Notebook used for **Exploratory Data Analysis (EDA)**.
Includes descriptive statistics, distribution visualization, correlation analysis, feature relevance inspection, and early insights guiding modeling choices.

#### [main.ipynb](src/main.ipynb)

Contains some old analysis abouts regression and claissification models. 

#### [regression.ipynb](src/regression.ipynb)

Notebook focused on **predicting wine quality as a regression problem**.
Covers model design, evaluation with regression metrics, comparison across algorithms, and interpretation of predictive behavior.





## Installation

### Working Environment
- **Recommended IDE**: VS-Code
- **AI Code Autocompletion**: GitHub Copilot 
- **AI Code Assistance**: Claude Sonnet 4.5, ChatGPT 5.1 
- **Python Version**: 3.10.16
- **Environment Manager**: Conda

#### Create Virtual Environment
```bash
conda create -n {name} python=3.10.16
conda activate {name}
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```


## External Elements / Citations

- Wine Type Classification Dataset, Kaggle Access [Here](https://www.kaggle.com/datasets/ehsanesmaeili/red-and-white-wine-quality-merged)
- Copyright (c) 2013 Mark Otto.
- Copyright (c) 2017 Andrew Fong.


## Contact / Support / Author

For questions or issues regarding code execution, contact(s):
- Mathieu Adnet: 
- Colin MANYRI: colin.manyri@etu.utc.fr

