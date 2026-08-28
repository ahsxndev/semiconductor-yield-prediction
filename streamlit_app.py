"""
Semiconductor Yield Prediction Dashboard
An interactive Streamlit app for wafer pass/fail prediction using Random Forest.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.model_utils import load_engineered_data

# Page configuration
st.set_page_config(
    page_title="SECOM Yield Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = project_root / "models" / "best_rf_model.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_test_data():
    """Load test dataset."""
    data_dir = str(project_root / "data" / "engineered")
    _, _, _, _, X_test, y_test = load_engineered_data(use_smote=False, use_pca=False, data_dir=data_dir)
    return X_test, y_test


@st.cache_data
def load_model_report():
    """Load final model report."""
    report_path = project_root / "reports" / "final_model_report.json"
    with open(report_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_model_comparison():
    """Load model comparison CSV."""
    csv_path = project_root / "reports" / "model_comparison.csv"
    return pd.read_csv(csv_path)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Model Performance", "🔮 Make Predictions", "🔍 Feature Analysis", "ℹ️ About"]
)

# Load data and model
try:
    model = load_model()
    X_test, y_test = load_test_data()
    report = load_model_report()
    model_comparison = load_model_comparison()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🔬 Semiconductor Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown("### Predicting Wafer Pass/Fail Using Machine Learning")

    st.markdown("---")

    # Project overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📋 Project Overview")
        st.markdown("""
        This machine learning system predicts semiconductor wafer pass/fail outcomes based on
        sensor measurements from the manufacturing process. The model helps identify potential
        failures early, reducing costs and improving production efficiency.

        **Key Features:**
        - 🎯 **High Precision**: 75% precision - minimal false alarms
        - 🔍 **Interpretable**: SHAP analysis shows which features drive predictions
        - ⚡ **Fast**: Real-time predictions in milliseconds
        - 📈 **Robust**: Handles extreme class imbalance and missing data
        """)

        st.markdown("#### 🎯 Business Impact")
        st.markdown("""
        - **Cost Reduction**: Early failure detection prevents downstream waste
        - **Quality Improvement**: Data-driven insights for process optimization
        - **Efficiency**: Automated screening reduces manual inspection time
        """)

    with col2:
        st.markdown("#### 📊 Dataset")
        st.info(f"""
        **Samples**: {report['dataset']['total_samples']:,}
        **Features**: {report['dataset']['features_original']} → {report['dataset']['features_engineered']}
        **Reduction**: {report['dataset']['feature_reduction']}
        **Failure Rate**: {report['dataset']['failure_rate']}
        **Imbalance**: {report['dataset']['class_imbalance_ratio']}
        """)

        st.markdown("#### 🏆 Best Model")
        st.success(f"""
        **{report['best_model']['name']}**

        **Test Performance**:
        - F1 Score: {report['best_model']['test_performance']['f1_score']:.4f}
        - Precision: {report['best_model']['test_performance']['precision']:.2%}
        - Recall: {report['best_model']['test_performance']['recall']:.2%}
        - ROC-AUC: {report['best_model']['test_performance']['roc_auc']:.4f}
        """)

    st.markdown("---")

    # Key Metrics
    st.markdown("#### 📈 Key Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{report['best_model']['test_performance']['accuracy']:.2%}",
            help="Overall correct predictions"
        )

    with col2:
        st.metric(
            "Precision",
            f"{report['best_model']['test_performance']['precision']:.2%}",
            help="When predicting Fail, how often is it correct?"
        )

    with col3:
        st.metric(
            "Recall",
            f"{report['best_model']['test_performance']['recall']:.2%}",
            help="Of all actual failures, how many did we catch?"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{report['best_model']['test_performance']['f1_score']:.4f}",
            help="Harmonic mean of precision and recall"
        )

    st.markdown("---")

    # Confusion Matrix Summary
    st.markdown("#### 🎯 Test Set Results")

    cm_data = report['best_model']['confusion_matrix']
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            [[cm_data['true_negatives'], cm_data['false_positives']],
             [cm_data['false_negatives'], cm_data['true_positives']]],
            index=['Actual Pass', 'Actual Fail'],
            columns=['Predicted Pass', 'Predicted Fail']
        )
        st.dataframe(cm_df, use_container_width=True)

    with col2:
        st.markdown("**Interpretation**")
        st.markdown(f"""
        - ✅ **Failures Caught**: {cm_data['failures_caught']}
        - ⚠️ **Failures Missed**: {cm_data['failures_missed']}
        - 🔔 **False Alarms**: {cm_data['false_positives']} out of {cm_data['true_negatives'] + cm_data['false_positives']} passing wafers
        - 🎯 **True Negatives**: {cm_data['true_negatives']} correctly identified as Pass
        """)

    st.markdown("---")

    # Quick start
    st.markdown("#### 🚀 Get Started")
    st.markdown("""
    1. **📊 Model Performance**: View detailed performance metrics and visualizations
    2. **🔮 Make Predictions**: Upload your data or use sample data for predictions
    3. **🔍 Feature Analysis**: Explore feature importance and SHAP analysis
    4. **ℹ️ About**: Learn about the methodology and technical details
    """)

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrix", "ROC & PR Curves"])

    with tab1:
        st.markdown("### Model Comparison")
        st.markdown("Compare performance across all evaluated models")

        # Model comparison table
        st.dataframe(
            model_comparison.style.highlight_max(
                subset=['test_f1', 'test_precision', 'test_recall', 'test_roc_auc'],
                color='lightgreen'
            ),
            use_container_width=True
        )

        # Interactive comparison chart
        metrics_to_plot = ['test_f1', 'test_precision', 'test_recall', 'test_roc_auc']
        metric_names = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']

        selected_metric = st.selectbox("Select metric to visualize:", metric_names)
        metric_col = metrics_to_plot[metric_names.index(selected_metric)]

        fig = px.bar(
            model_comparison.sort_values(metric_col, ascending=True),
            x=metric_col,
            y='model',
            orientation='h',
            color='type',
            title=f'Model Comparison - {selected_metric}',
            labels={metric_col: selected_metric, 'model': 'Model'},
            color_discrete_map={'Baseline': 'steelblue', 'Advanced': 'darkgreen'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Confusion Matrix - Test Set")

        # Calculate confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Pass', 'Predicted Fail'],
            y=['Actual Pass', 'Actual Fail'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            hoverongaps=False
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics breakdown
        col1, col2, col3, col4 = st.columns(4)

        tn, fp, fn, tp = cm.ravel()

        with col1:
            st.metric("True Negatives", tn, help="Correctly predicted Pass")
        with col2:
            st.metric("False Positives", fp, help="Pass predicted as Fail", delta=f"-{fp}", delta_color="inverse")
        with col3:
            st.metric("False Negatives", fn, help="Fail predicted as Pass", delta=f"-{fn}", delta_color="inverse")
        with col4:
            st.metric("True Positives", tp, help="Correctly predicted Fail")

    with tab3:
        st.markdown("### ROC & Precision-Recall Curves")

        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        y_proba = model.predict_proba(X_test)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve')
        )

        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})',
                      line=dict(color='darkgreen', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='black', dash='dash')),
            row=1, col=1
        )

        # PR Curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC={pr_auc:.3f})',
                      line=dict(color='darkgreen', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[y_test.mean(), y_test.mean()], mode='lines',
                      name=f'Baseline ({y_test.mean():.3f})',
                      line=dict(color='black', dash='dash')),
            row=1, col=2
        )

        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)

        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAKE PREDICTIONS PAGE
# ============================================================================
elif page == "🔮 Make Predictions":
    st.markdown('<p class="main-header">🔮 Make Predictions</p>', unsafe_allow_html=True)

    st.markdown("""
    Upload your preprocessed data or use sample test data to get predictions.

    **Note**: Data must be preprocessed using the same feature engineering pipeline
    (100 features after engineering).
    """)

    st.markdown("---")

    # Prediction options
    pred_option = st.radio(
        "Choose prediction method:",
        ["Use Sample Test Data", "Upload CSV File"]
    )

    if pred_option == "Use Sample Test Data":
        st.info("Using random samples from the test set")

        # Number of samples
        n_samples = st.slider("Number of samples to predict:", 1, 20, 5)

        # Random sample
        if st.button("Generate Random Samples"):
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
            y_true = y_test.iloc[sample_indices]

            # Make predictions
            predictions = model.predict(X_sample)
            probabilities = model.predict_proba(X_sample)[:, 1]

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Sample': range(1, n_samples + 1),
                'True Label': ['Pass' if y == 0 else 'Fail' for y in y_true],
                'Predicted Label': ['Pass' if p == 0 else 'Fail' for p in predictions],
                'Failure Probability': [f'{p:.2%}' for p in probabilities],
                'Correct': ['✅' if t == p else '❌' for t, p in zip(y_true, predictions)]
            })

            st.markdown("### Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # Accuracy on sample
            accuracy = (y_true == predictions).mean()
            st.success(f"**Accuracy on sample**: {accuracy:.2%} ({(y_true == predictions).sum()}/{n_samples} correct)")

    else:
        st.info("Upload a CSV file with preprocessed features (100 columns)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                X_upload = pd.read_csv(uploaded_file)

                # Validate shape
                if X_upload.shape[1] != 100:
                    st.error(f"Expected 100 features, got {X_upload.shape[1]}. Please ensure data is preprocessed.")
                else:
                    st.success(f"Loaded {len(X_upload)} samples with {X_upload.shape[1]} features")

                    # Show preview
                    with st.expander("Preview data"):
                        st.dataframe(X_upload.head())

                    if st.button("Make Predictions"):
                        # Make predictions
                        predictions = model.predict(X_upload)
                        probabilities = model.predict_proba(X_upload)[:, 1]

                        # Create results
                        results_df = pd.DataFrame({
                            'Sample': range(1, len(X_upload) + 1),
                            'Predicted Label': ['Pass' if p == 0 else 'Fail' for p in predictions],
                            'Failure Probability': probabilities,
                            'Confidence': [max(1-p, p) for p in probabilities]
                        })

                        st.markdown("### Prediction Results")
                        st.dataframe(results_df, use_container_width=True)

                        # Summary
                        n_fails = (predictions == 1).sum()
                        n_pass = (predictions == 0).sum()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(X_upload))
                        with col2:
                            st.metric("Predicted Pass", n_pass)
                        with col3:
                            st.metric("Predicted Fail", n_fails)

                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Predictions",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

# ============================================================================
# FEATURE ANALYSIS PAGE
# ============================================================================
elif page == "🔍 Feature Analysis":
    st.markdown('<p class="main-header">🔍 Feature Analysis</p>', unsafe_allow_html=True)

    st.markdown("### Feature Importance Analysis")
    st.markdown("Understanding which features drive the model's predictions")

    tab1, tab2 = st.tabs(["Feature Importance", "SHAP Analysis"])

    with tab1:
        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Top N slider
        top_n = st.slider("Number of top features to display:", 10, 50, 20)

        # Plot
        fig = px.bar(
            feature_importances.head(top_n).sort_values('Importance'),
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance table
        with st.expander("View all feature importances"):
            st.dataframe(feature_importances, use_container_width=True)

    with tab2:
        st.markdown("### SHAP (SHapley Additive exPlanations)")
        st.markdown("""
        SHAP values show how each feature contributes to individual predictions.
        Computing SHAP values for visualization...
        """)

        with st.spinner("Calculating SHAP values (this may take a moment)..."):
            # Sample for SHAP (to speed up computation)
            n_shap_samples = min(100, len(X_test))
            X_shap = X_test.sample(n_shap_samples, random_state=42)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            # If binary classification, select positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # SHAP summary plot
            st.markdown("#### SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=False)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **Interpretation**:
            - Higher SHAP values indicate features that have more impact on model predictions
            - These are the key sensor measurements driving failure detection
            """)

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Methodology", "Model Details", "Usage Guide"])

    with tab1:
        st.markdown("### Methodology")

        st.markdown("#### 1. Data Preprocessing")
        st.markdown("""
        - **Missing Value Handling**: Adaptive imputation strategy
          - <5% missing: Simple imputation (median)
          - 5-20%: KNN imputation (k=5)
          - 20-40%: Iterative imputation
          - >40%: Remove feature
        - **Normalization**: Yeo-Johnson power transformation for 222 skewed features
        - **Scaling**: StandardScaler for all features
        """)

        st.markdown("#### 2. Feature Engineering")
        st.markdown(f"""
        - Removed 148 high-missing and zero-variance features
        - Selected top 100 features using ANOVA F-test (SelectKBest)
        - **Dimensionality Reduction**: {report['dataset']['feature_reduction']}
          ({report['dataset']['features_original']} → {report['dataset']['features_engineered']} features)
        """)

        st.markdown("#### 3. Class Imbalance Handling")
        st.markdown("""
        - Tested SMOTE (synthetic oversampling) vs class_weight='balanced'
        - **Selected**: class_weight='balanced' - better generalization
        - SMOTE caused overfitting to synthetic data
        """)

        st.markdown("#### 4. Model Selection")
        st.markdown("""
        Models evaluated:
        - Dummy Classifier (baseline)
        - Logistic Regression
        - Decision Tree
        - Gaussian Naive Bayes
        - **Random Forest** (selected)

        Random Forest selected for:
        - Highest F1 score and precision
        - Robust to outliers and non-linear patterns
        - Interpretable via feature importance and SHAP
        """)

    with tab2:
        st.markdown("### Model Details")

        st.markdown("#### Random Forest Hyperparameters")
        params = report['best_model']['parameters']
        params_df = pd.DataFrame({
            'Parameter': params.keys(),
            'Value': params.values()
        })
        st.table(params_df)

        st.markdown("#### Performance Metrics")
        perf = report['best_model']['test_performance']
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC'],
            'Value': [
                f"{perf['accuracy']:.2%}",
                f"{perf['precision']:.2%}",
                f"{perf['recall']:.2%}",
                f"{perf['f1_score']:.4f}",
                f"{perf['roc_auc']:.4f}",
                f"{perf['pr_auc']:.4f}"
            ]
        })
        st.table(metrics_df)

        st.markdown("#### Model Strengths & Limitations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Strengths**")
            st.markdown("""
            - High precision (75%) - minimal false alarms
            - Best F1 score among all models
            - Handles non-linear relationships
            - Robust to outliers
            - Interpretable results
            """)

        with col2:
            st.markdown("**⚠️ Limitations**")
            st.markdown("""
            - Conservative recall (14.29%)
            - Misses some failures
            - Limited by training data quality
            - Requires preprocessed input
            """)

    with tab3:
        st.markdown("### Usage Guide")

        st.markdown("#### For Predictions")
        st.code("""
import joblib
import pandas as pd

# Load model
model = joblib.load('models/best_rf_model.pkl')

# Load preprocessed data (100 features)
X_new = pd.read_csv('preprocessed_data.csv')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Interpret results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "Fail" if pred == 1 else "Pass"
    print(f"Sample {i+1}: {label} (Probability: {prob:.2%})")
        """, language='python')

        st.markdown("#### Data Requirements")
        st.markdown("""
        - Data must be preprocessed using the feature engineering pipeline
        - Requires exactly 100 features (after engineering)
        - Features must be in the same order as training data
        - Missing values should be imputed beforehand
        """)

        st.markdown("#### Interpretation Tips")
        st.markdown("""
        - **High failure probability (>50%)**: Likely to fail - investigate immediately
        - **Medium probability (20-50%)**: Borderline - consider secondary checks
        - **Low probability (<20%)**: Likely to pass - routine monitoring
        - **Precision is high**: When model predicts Fail, it's usually correct
        - **Recall is moderate**: Model may miss some failures - combine with other QC measures
        """)

    st.markdown("---")
    st.markdown("### Technologies Used")
    st.markdown("""
    - **ML Framework**: scikit-learn 1.6.1
    - **Data Processing**: Pandas 2.3.0, NumPy 2.3.0
    - **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
    - **Interpretability**: SHAP 0.47.0
    - **Tracking**: MLflow 2.19.0
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**SECOM Yield Prediction**

An ML-powered system for semiconductor wafer quality prediction.

**Dataset**: UCI SECOM
**Model**: Random Forest
**Performance**: F1=0.24, Precision=75%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown("""
- [GitHub Repository](#)
- [Project Documentation](README.md)
- [Model Report](reports/final_model_report.json)
""")
