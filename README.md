# Semiconductor Yield Prediction and Root Cause Analysis

### Cost-Sensitive Machine Learning with SHAP and LIME

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahsxndev/semiconductor-yield-prediction/blob/main/Semiconductor_Yield_Prediction_Research.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([YOUR_STREAMLIT_LINK](https://secom-yield-ai.streamlit.app/))

## Overview

Semiconductor manufacturing involves hundreds of sensor measurements collected during the fabrication process. Even a small number of defective chips can result in significant costs when defects are not detected before shipment.

This project explores a machine learning approach for predicting semiconductor manufacturing failures using the UCI SECOM dataset.

The main focus is not only on classification accuracy, but also on the cost of missing defective chips and understanding why a model makes a particular prediction.

The project compares Random Forest and XGBoost models and combines them with:

- Missing-value handling
- Feature selection
- SMOTE oversampling
- Cost-sensitive learning
- F-Beta evaluation
- Manufacturing cost analysis
- SHAP explanations
- LIME explanations
- Threshold optimization
- 5-fold cross-validation
- Interactive Streamlit inference

## Research Question

> Can a cost-sensitive machine learning pipeline improve the detection of defective semiconductor samples while providing explanations that can support root-cause analysis?

A second question is:

> Which sensor measurements have the greatest influence on the model's predictions, both globally and for an individual chip?

## Dataset

The project uses the **SECOM dataset** from the UCI Machine Learning Repository.

The dataset contains:

| Property | Value |
| --- | ---: |
| Observations | 1,567 |
| Sensor features | 590 |
| Task | Binary classification |
| Positive class | Fail |
| Negative class | Pass |

The dataset contains a large number of sensor measurements collected during semiconductor fabrication.

One of the main challenges is the strong class imbalance. Approximately 93% of the samples are Pass and only about 7% are Fail.

This makes accuracy an unreliable metric by itself. A model could achieve more than 93% accuracy simply by predicting Pass for almost every sample while failing to detect defective chips.

For this reason, the project gives more attention to Recall, F-Beta, PR-AUC and the cost associated with false negatives.

## Problem Formulation

The classification target is converted to:

```text
0 → Pass
1 → Fail
````

The main error of concern is a false negative:

```text
Actual: Fail
Predicted: Pass
```

In a manufacturing setting, this represents a defective chip that could pass inspection.

A false positive has a different consequence:

```text
Actual: Pass
Predicted: Fail
```

This may result in additional testing of a good chip.

Because these two errors do not have the same practical impact, the project treats the classification problem as cost-sensitive rather than optimizing accuracy alone.

## Methodology

The complete workflow is:

```text
SECOM Dataset
      ↓
Exploratory Data Analysis
      ↓
Missing Value Analysis
      ↓
Train/Test Split
      ↓
Feature Cleaning
      ↓
Tiered Imputation
      ↓
Yeo-Johnson Transformation
      ↓
Standard Scaling
      ↓
ANOVA Feature Selection
      ↓
SMOTE
      ↓
Random Forest
      ↓
XGBoost
      ↓
Model Evaluation
      ↓
Cost Analysis
      ↓
Threshold Optimization
      ↓
SHAP + LIME
      ↓
Streamlit Application
```

## Data Preprocessing

The preprocessing pipeline was designed around the characteristics of the sensor data.

### 1. Train/Test Split

An 80/20 stratified split is used while preserving the original class distribution.

### 2. Missing Values

Features with more than 40% missing values are removed.

For the remaining features, different imputation strategies are applied depending on their missing-value rate:

| Missing rate | Method               |
| ------------ | -------------------- |
| < 5%         | Median imputation    |
| 5% to 20%    | KNN imputation       |
| 20% to 40%   | Iterative imputation |

This avoids treating all missing-value patterns in exactly the same way.

### 3. Zero-Variance Features

Features with no variation in the training data are removed because they do not provide useful information for classification.

### 4. Yeo-Johnson Transformation

Highly skewed features are transformed using the Yeo-Johnson transformation.

### 5. Standardization

The remaining features are standardized using `StandardScaler`.

### 6. Feature Selection

The top 100 features are selected using the ANOVA F-test.

### 7. SMOTE

SMOTE is applied only to the training set to address the strong imbalance between Pass and Fail samples.

## Models

### Random Forest

Random Forest is used as the baseline model.

The classifier uses balanced class weights to give more importance to the minority Fail class.

```text
Random Forest
├── 300 trees
├── max_depth = 15
├── min_samples_split = 5
├── min_samples_leaf = 2
└── class_weight = balanced
```

### XGBoost

XGBoost is used as the main boosted-tree model.

The model uses:

* 500 estimators
* Learning rate of 0.05
* Maximum depth of 6
* Row and feature subsampling
* Regularization
* Class weighting
* PR-AUC as the evaluation metric

The class imbalance is addressed using both SMOTE and `scale_pos_weight`.

## Evaluation Strategy

Several metrics are used instead of relying on accuracy alone.

### F-Beta

The primary metric is:

```text
F-Beta (β = 2)
```

With β = 2, recall receives greater importance than precision.

This is appropriate for the project because missing a defective sample is considered more serious than unnecessarily testing a good sample.

Other metrics include:

* Recall
* Precision
* F1-score
* ROC-AUC
* PR-AUC

## Initial Model Results

The initial test-set results were:

| Metric       | Random Forest | XGBoost |
| ------------ | ------------: | ------: |
| F-Beta (β=2) |        0.0568 |  0.2500 |
| Recall       |        0.0476 |  0.2381 |
| Precision    |        0.2500 |  0.3125 |
| ROC-AUC      |        0.7052 |  0.6832 |
| PR-AUC       |        0.2183 |  0.2079 |
| Accuracy     |          0.93 |    0.91 |

The results show an important characteristic of the problem.

Although the overall accuracy is relatively high, the models have difficulty detecting the minority Fail class at the default classification threshold.

This is one of the reasons accuracy alone would give a misleading impression of model performance.

## Cost-Sensitive Evaluation

To examine the practical effect of classification errors, a simple manufacturing cost model was introduced.

The assumed costs were:

| Error          | Assumed cost |
| -------------- | -----------: |
| False Negative |      $10,000 |
| False Positive |         $100 |

Under these assumptions, missing one defective chip is considerably more expensive than unnecessarily re-testing a good chip.

Using the default threshold:

| Model         | False Negatives | False Positives | Estimated Cost |
| ------------- | --------------: | --------------: | -------------: |
| Random Forest |              20 |               3 |       $200,300 |
| XGBoost       |              16 |              11 |       $161,100 |

Under this cost assumption, XGBoost reduces the estimated cost by:

```text
$39,200 per test batch
```

The purpose of this calculation is to demonstrate how model selection can change when the cost of different errors is taken into account.

The dollar values are assumptions for the experiment and should not be interpreted as actual semiconductor industry costs.

## Threshold Optimization

The default classification threshold of 0.5 is not necessarily appropriate for a highly imbalanced and cost-sensitive problem.

The project therefore evaluates different probability thresholds and selects the threshold that minimizes the defined manufacturing cost.

The selected threshold for the final XGBoost evaluation was:

```text
Threshold = 0.10
```

At this threshold:

| Metric          |   Result |
| --------------- | -------: |
| F-Beta (β=2)    |   0.4135 |
| Recall          |   0.5238 |
| Precision       |   0.2245 |
| False Negatives |       10 |
| False Positives |       38 |
| Estimated Cost  | $103,800 |

Compared with the default threshold, the optimized threshold detects more defective samples while accepting a higher number of false alarms.

This is an example of the trade-off between recall and precision in a cost-sensitive inspection system.

## Cross-Validation

The XGBoost model was also evaluated using 5-fold stratified cross-validation on the resampled training data.

The F-Beta scores across the five folds were:

```text
0.9806
0.9831
0.9915
0.9907
0.9799
```

Mean:

```text
0.9852
```

Standard deviation:

```text
0.0050
```

The low variation across the folds indicates stable performance under the cross-validation setup.

These scores should be interpreted separately from the held-out test-set results because the cross-validation was performed on the SMOTE-resampled training data.

## Explainability

A prediction is more useful in a manufacturing environment when engineers can investigate why it was produced.

This project therefore uses two complementary explainability methods.

### SHAP

SHAP is used for global feature analysis.

The XGBoost model is explained using `TreeExplainer` to identify which sensor features have the greatest influence on predictions across the test set.

The SHAP analysis helps answer:

> Which sensor measurements are most influential in distinguishing Pass and Fail samples?

### LIME

LIME is used for local explanations.

For an individual chip, LIME creates a local interpretable approximation of the model and identifies the sensor conditions that contributed to the prediction.

This allows a prediction such as:

```text
Chip #0
Prediction: Pass
Failure risk: 0.3%
```

to be investigated at the individual-sample level.

## SHAP vs LIME

The project compares both approaches:

| Aspect              | SHAP                           | LIME                               |
| ------------------- | ------------------------------ | ---------------------------------- |
| Main purpose        | Global and local explanations  | Local explanations                 |
| Model dependency    | TreeExplainer used for XGBoost | Model-agnostic                     |
| Output              | Feature contribution values    | Local feature rules                |
| Use in this project | Identify important sensors     | Investigate individual predictions |

The project also compares the top features identified by both methods for the same sample.

This provides a simple way to examine whether the two explanation methods identify similar factors.

## Streamlit Application

The trained model was integrated into a Streamlit application for interactive inspection.

The interface allows a user to select a processed chip from the test set and inspect:

* Chip ID
* Model prediction
* Actual ground truth
* Failure probability
* LIME-based local explanation
* Important sensor features

### Example

```text
Selected Chip: #0

Prediction:
Pass

Actual:
Pass

Failure Risk:
0.3%
```

The application is intended as a demonstration of how an ML model and its explanations could be presented to a quality-control engineer.

It is not a production semiconductor inspection system.

## Project Structure

```text
semiconductor-yield-prediction/
│
├── data/
│   └── SECOM dataset
│
├── models/
│   ├── xgb_model.json
│   ├── best_threshold.joblib
│   ├── X_test_preprocessed.csv
│   ├── y_test.csv
│   └── X_train_resampled.csv
│
├── Semiconductor_Yield_Prediction_Research.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Technologies

### Machine Learning

* Python
* Scikit-learn
* XGBoost
* Random Forest
* SMOTE

### Explainable AI

* SHAP
* LIME

### Data Processing

* NumPy
* Pandas
* Scikit-learn preprocessing

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit
* Google Colab
* Jupyter Notebook

## Reproducibility

The complete experimental workflow is available in the Jupyter notebook.

The notebook covers:

1. Dataset loading
2. Exploratory analysis
3. Missing-value analysis
4. Preprocessing
5. Feature selection
6. SMOTE
7. Random Forest training
8. XGBoost training
9. Model evaluation
10. Cost analysis
11. Cross-validation
12. SHAP analysis
13. LIME analysis
14. Threshold optimization
15. Model export

The notebook can be opened and executed in Google Colab.

## Limitations

There are several limitations to this study.

First, the dataset is relatively small compared with the number of sensor features. The high-dimensional feature space makes feature selection and preprocessing important.

Second, the manufacturing cost values used in the cost matrix are assumptions for experimentation. Actual costs would need to be obtained from a real manufacturing environment.

Third, the test set contains a relatively small number of Fail samples. As a result, recall and precision can change substantially with a small number of classification changes.

Finally, the project evaluates the model on the SECOM benchmark dataset. Performance on a different semiconductor manufacturing process cannot be assumed to be the same.

## Future Work

Several extensions could make the system more useful for real manufacturing environments:

* Evaluate temporal drift in sensor distributions
* Explore unsupervised anomaly detection
* Perform systematic hyperparameter optimization
* Investigate alternative feature-selection methods
* Compare additional gradient boosting models
* Calibrate predicted probabilities
* Add model monitoring
* Investigate MLflow for model versioning
* Evaluate the system on additional semiconductor datasets
* Integrate real manufacturing cost information

## Conclusion

This project explores semiconductor yield prediction as a cost-sensitive classification problem rather than treating it as a simple accuracy-based prediction task.

The experiments show that the choice of decision threshold can substantially change the balance between missed defects and unnecessary re-tests.

XGBoost provided better defect detection than the Random Forest baseline under the default evaluation, while threshold optimization further increased recall and reduced the estimated manufacturing cost under the assumptions used in this study.

The addition of SHAP and LIME provides two different views of model behavior. SHAP helps identify influential sensors across the dataset, while LIME provides a local explanation for an individual prediction.

Together, the modeling, cost analysis and explainability components provide a small end-to-end example of how machine learning can be used for semiconductor quality analysis.

## References

* UCI Machine Learning Repository, SECOM Dataset
* Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System
* Lundberg, S. M. & Lee, S. I. A Unified Approach to Interpreting Model Predictions
* Ribeiro, M. T., Singh, S. & Guestrin, C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier
* Chawla, N. V. et al. SMOTE: Synthetic Minority Over-sampling Technique

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Ahsan Zaman**

Computer Science Student | Machine Learning | Deep Learning | Data Science

GitHub: [https://github.com/ahsxndev](https://github.com/ahsxndev)
LinkedIn: [https://www.linkedin.com/in/ahxanzaman/](https://www.linkedin.com/in/ahxanzaman/)
