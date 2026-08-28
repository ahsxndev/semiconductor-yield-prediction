# Semiconductor Manufacturing Yield Prediction

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Semiconductor Yield Prediction

An interactive Streamlit application for predicting semiconductor wafer yield from processed SECOM sensor data. The dashboard lets you inspect a test-set chip, view its predicted failure risk, compare the prediction with the ground truth, and explore a local LIME explanation.

## Features

- XGBoost binary classification for Pass/Fail prediction
- Configurable failure threshold loaded from the trained model assets
- Interactive test-chip selection in the Streamlit sidebar
- Failure probability and predicted label
- Ground-truth comparison for test-set samples
- LIME-based local root-cause explanation with the ten most influential features

## Project Structure

```text
semiconductor-yield-prediction/
├── app.py                                  # Streamlit dashboard
├── requirements.txt                         # Python dependencies
├── Semiconductor_Yield_Prediction_Research.ipynb
│                                            # Research and model-development notebook
└── models/
  ├── xgb_model.json                       # XGBoost model loaded by the app
  ├── xgb_model.joblib                     # Serialized model artifact
  ├── best_threshold.joblib                # Selected classification threshold
  ├── X_test_preprocessed.csv              # Preprocessed test features
  ├── X_train_resampled.csv               # Training data used by LIME
  └── y_test.csv                            # Test-set labels
```

## Requirements

- Python 3.10 or newer
- pip
- The files in the `models/` directory

The dependency versions are defined in [requirements.txt](requirements.txt). The project includes XGBoost, scikit-learn, pandas, NumPy, Streamlit, Matplotlib, and LIME, along with the notebook and analysis dependencies.

## Installation

Open a terminal in the project directory and run:

```cmd
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

To install the packages directly instead, run:

```cmd
py -m pip install streamlit pandas numpy joblib xgboost lime matplotlib
```

## Run the Dashboard

```cmd
py -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`.

Use the **Batch Inspection** slider to select a chip from the preprocessed test set. The dashboard then calculates the failure probability, applies the saved threshold, displays the actual label, and generates a LIME explanation.

## Model and Data Notes

- The application expects all paths relative to the project root.
- `xgb_model.json` is the native XGBoost model loaded by `app.py`.
- `best_threshold.joblib` controls whether a sample is classified as Pass or Fail.
- `X_test_preprocessed.csv` must have the same feature columns and ordering used during model training.
- `X_train_resampled.csv` supplies the reference data and feature names for the LIME explainer.
- The app uses the labels in `y_test.csv` only to show the known outcome for the selected test sample.

## Research Notebook

Open [Semiconductor_Yield_Prediction_Research.ipynb](Semiconductor_Yield_Prediction_Research.ipynb) to review the data exploration, preprocessing, resampling, model development, and evaluation work behind the saved artifacts.

Start Jupyter with:

```cmd
py -m jupyter notebook Semiconductor_Yield_Prediction_Research.ipynb
```

## Dataset

The project is based on the [UCI SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM), which contains semiconductor manufacturing sensor measurements and a binary pass/fail target. The dashboard consumes the processed model inputs rather than the raw dataset.

## License

This project is licensed under the MIT License.

## Author

**Adhokshaj Baliga**

- [LinkedIn](https://www.linkedin.com/in/adhokshaj1/)
- [GitHub](https://github.com/Adhokshaj04)

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Author](#author)

## Problem Statement

Semiconductor manufacturing is a complex process where wafer failures can result in significant financial losses. The challenge is to:
- Predict wafer pass/fail outcomes based on 590 sensor measurements
- Handle extreme class imbalance (93.3% pass, 6.7% fail)
- Deal with significant missing data (8% overall, some features >40%)
- Provide interpretable results for process improvement

**Business Impact**: Early detection of potential failures enables:
- Reduced manufacturing costs through early intervention
- Improved yield rates and production efficiency
- Data-driven process optimization insights

## Dataset

**Source**: [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)

**Characteristics**:
- **Samples**: 1,567 wafers
- **Features**: 590 sensor measurements (continuous)
- **Target**: Binary (Pass=0, Fail=1)
- **Class Distribution**: 1,463 Pass (93.3%) | 104 Fail (6.7%)
- **Imbalance Ratio**: 1:14
- **Missing Values**: ~8% overall, with some features having >40% missing

**Challenges**:
- Extreme class imbalance requiring specialized techniques
- High dimensionality (curse of dimensionality)
- Significant missing data requiring robust imputation
- Limited failure examples for model training

## Project Structure

```
semiconductor-yield-prediction/
│
├── data/
│   ├── raw/                    # Original SECOM dataset
│   ├── processed/              # Train/val/test splits
│   └── engineered/             # Feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # EDA and statistical analysis
│   ├── 02_feature_engineering.ipynb        # Feature engineering pipeline
│   ├── 03_baseline_models.ipynb            # Baseline model comparison
│   ├── 04_advanced_models_rf_only.ipynb    # Random Forest tuning
│   └── 05_final_evaluation.ipynb           # Comprehensive model comparison
│
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data loading utilities
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering class
│   └── models/
│       └── model_utils.py      # Model evaluation & tracking
│
├── models/
│   └── best_rf_model.pkl       # Trained Random Forest model
│
├── reports/
│   ├── figures/                # Visualizations and plots
│   ├── final_model_report.json # Comprehensive results
│   └── model_comparison.csv    # All model metrics
│
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites
- Python 3.13+
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Adhokshaj04/semiconductor-yield-prediction.git
cd semiconductor-yield-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the SECOM dataset
# Download from: https://archive.ics.uci.edu/ml/datasets/SECOM
# Or use the provided script:
# Place uci-secom.csv in data/raw/
```

### Dependencies

Key packages:
- `scikit-learn==1.6.1` - Machine learning models
- `pandas==2.3.0` - Data manipulation
- `numpy==2.3.0` - Numerical computing
- `matplotlib==3.10.0` - Visualization
- `seaborn==0.13.2` - Statistical visualization
- `mlflow==2.19.0` - Experiment tracking
- `shap==0.47.0` - Model interpretability
- `imbalanced-learn==0.12.4` - SMOTE and sampling techniques

See [requirements.txt](requirements.txt) for complete list.

## Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Performs comprehensive EDA including:
- Missing value analysis
- Feature distributions and correlations
- Pass vs Fail statistical comparisons
- Outlier detection and feature variance analysis

### 2. Feature Engineering

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

Transforms raw data through:
- Removes 148 features (high missing + zero variance)
- Adaptive imputation (Simple/KNN/Iterative based on missing %)
- Power transformation for 222 highly skewed features
- Feature selection (SelectKBest) reducing to 100 features
- **Result**: 590 → 100 features (83% reduction)

### 3. Model Training

```bash
# Baseline models
jupyter notebook notebooks/03_baseline_models.ipynb

# Advanced models (Random Forest)
jupyter notebook notebooks/04_advanced_models_rf_only.ipynb
```

### 4. Final Evaluation

```bash
jupyter notebook notebooks/05_final_evaluation.ipynb
```

Generates comprehensive model comparison and final results.

### 5. Making Predictions

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/best_rf_model.pkl')

# Load your preprocessed data
X_new = pd.read_csv('your_preprocessed_data.csv')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

print(f"Predicted class: {predictions[0]}")
print(f"Failure probability: {probabilities[0]:.2%}")
```

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Adaptive imputation strategy
  - <5% missing: Simple imputation (median)
  - 5-20%: KNN imputation (k=5)
  - 20-40%: Iterative imputation
  - >40%: Remove feature
- **Normalization**: Yeo-Johnson power transformation for skewed features
- **Scaling**: StandardScaler for all features

### 2. Feature Engineering
- Removed 148 high-missing and zero-variance features
- Selected top 100 features using ANOVA F-test (SelectKBest)
- **Dimensionality Reduction**: 83% reduction (590 → 100 features)

### 3. Class Imbalance Handling
Compared two approaches:
- **SMOTE**: Synthetic oversampling (resulted in overfitting)
- **class_weight='balanced'**: ✅ Better generalization (selected)

### 4. Models Evaluated
| Model | Type | Approach |
|-------|------|----------|
| Dummy Classifier | Baseline | Most frequent class |
| Logistic Regression | Baseline | class_weight='balanced' |
| Decision Tree | Baseline | class_weight='balanced' |
| Gaussian Naive Bayes | Baseline | No balancing |
| **Random Forest** | **Advanced** | **class_weight + tuning** |

### 5. Evaluation Metrics
Given extreme class imbalance, prioritized:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Minimize false alarms (production cost)
- **Recall**: Catch actual failures (quality assurance)
- **ROC-AUC**: Overall discriminative ability

## Results

### Best Model: Random Forest (Tuned)

**Test Set Performance**:
```
Accuracy:   93.95%
Precision:  75.00%  ← High confidence predictions
Recall:     14.29%  ← Conservative (catches 3/21 failures)
F1 Score:   0.2400  ← Best among all models
ROC-AUC:    0.7096
PR-AUC:     0.2785
```

**Confusion Matrix** (Test Set):
```
                Predicted
              Pass    Fail
Actual Pass   292      1     ← 99.7% Pass correctly identified
      Fail     18      3     ← 14.3% Fail correctly caught
```

**Key Metrics**:
- Catches **3 out of 21 failures** with 75% precision
- Only **1 false alarm** out of 293 passing wafers
- **30.7% improvement** over baseline (Logistic Regression F1: 0.22)

### Model Comparison

| Model | F1 Score | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| **Random Forest (Tuned)** | **0.2400** | **0.7500** | 0.1429 | **0.7096** |
| Logistic Regression | 0.2222 | 0.1500 | **0.4286** | **0.7200** |
| Decision Tree | 0.1600 | 0.1379 | 0.1905 | 0.6000 |
| Naive Bayes | 0.1481 | 0.1212 | 0.1905 | 0.6700 |
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

### Hyperparameters (Best Random Forest)

```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'bootstrap': False,
    'class_weight': 'balanced'
}
```

Conservative parameters prevent overfitting while maintaining precision.

## Key Insights

### 1. Class Imbalance Strategy
- **class_weight='balanced'** outperformed SMOTE significantly
- SMOTE caused overfitting (CV F1: 0.99 but Val F1: 0.00)
- Real data with class weights generalizes better than synthetic data

### 2. Feature Importance (Top 5)
Based on SHAP analysis, most predictive features for failure detection:
1. Feature_26
2. Feature_42
3. Feature_78
4. Feature_15
5. Feature_91

*Note: Feature names are anonymized in the SECOM dataset*

### 3. Model Trade-offs

**Random Forest Strengths**:
- Highest precision (75%) - minimal false alarms
- Best F1 score overall
- Interpretable via SHAP and feature importance
- Robust to outliers and non-linear patterns

**Random Forest Limitations**:
- Lower recall (14.29%) than Logistic Regression (42.86%)
- Conservative - misses some failures
- Best suited when false alarm cost is high

### 4. Business Recommendations

**For High-Confidence Use Cases**:
- Deploy Random Forest when precision is critical
- Use when production disruption cost is high
- Combine with secondary quality checks

**For Maximum Coverage**:
- Consider Logistic Regression for higher recall
- Accept more false alarms to catch more failures
- Use ensemble voting (RF + LR) for balanced approach

### 5. Data Quality Impact
- Limited failure examples (104 total) constrains model performance
- **Recommendation**: Collect more failure data to improve minority class learning
- Missing values handled effectively through adaptive imputation

## Technologies Used

**Core ML Stack**:
- Python 3.13
- scikit-learn 1.6.1
- NumPy 2.3.0
- Pandas 2.3.0

**Visualization**:
- Matplotlib 3.10.0
- Seaborn 0.13.2
- SHAP 0.47.0

**Experiment Tracking**:
- MLflow 2.19.0

**Development Tools**:
- Jupyter Notebook
- Git version control

## Future Work

### Model Improvements
- [ ] Install XGBoost/LightGBM (resolve libomp dependency)
- [ ] Implement ensemble voting (Random Forest + Logistic Regression)
- [ ] Explore deep learning approaches (TabNet, FT-Transformer)
- [ ] Cost-sensitive learning with custom loss functions
- [ ] Threshold optimization using business cost matrix

### Deployment
- [ ] Streamlit dashboard for interactive predictions
- [ ] FastAPI REST endpoint for production integration
- [ ] Docker containerization
- [ ] MLOps pipeline with monitoring and retraining
- [ ] A/B testing framework

### Data Enhancement
- [ ] Collect more failure examples for better minority class learning
- [ ] Feature engineering based on domain expertise
- [ ] Time-series analysis if temporal patterns exist
- [ ] Semi-supervised learning with unlabeled data

### Interpretability
- [ ] LIME analysis for individual predictions
- [ ] Partial dependence plots
- [ ] Feature interaction analysis
- [ ] Process improvement recommendations based on SHAP

## Author

**Your Name**
- LinkedIn: [Adhokshaj Baliga](https://www.linkedin.com/in/adhokshaj1/)
- GitHub: [Adhokshaj Baliga](https://github.com/Adhokshaj04)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the SECOM dataset
- Skyworks Solutions for inspiration and job opportunity alignment
- Open-source community for excellent ML tools and libraries

---

**Last Updated**: December 2025