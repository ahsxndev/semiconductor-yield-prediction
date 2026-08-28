"""
Model training, evaluation, and tracking utilities for SECOM project.

This module provides utilities for:
- Model training and evaluation
- Performance metrics calculation
- MLflow experiment tracking
- Model comparison and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare machine learning models."""

    def __init__(self, experiment_name: str = "secom_yield_prediction"):
        """
        Initialize the model evaluator.

        Args:
            experiment_name: Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        self.results = []

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate a model and log results to MLflow.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model
            params: Model parameters

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_train_proba = y_train_pred
            y_val_proba = y_val_pred

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            # Training metrics
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            # Validation metrics
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
        }

        # Add AUC metrics if probabilities available
        if hasattr(model, 'predict_proba'):
            metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
            metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_proba)
            metrics['train_pr_auc'] = average_precision_score(y_train, y_train_proba)
            metrics['val_pr_auc'] = average_precision_score(y_val, y_val_proba)

        # Store results
        self.results.append(metrics)

        # Log to MLflow
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics({
                'train_accuracy': metrics['train_accuracy'],
                'train_precision': metrics['train_precision'],
                'train_recall': metrics['train_recall'],
                'train_f1': metrics['train_f1'],
                'val_accuracy': metrics['val_accuracy'],
                'val_precision': metrics['val_precision'],
                'val_recall': metrics['val_recall'],
                'val_f1': metrics['val_f1'],
            })

            if 'val_roc_auc' in metrics:
                mlflow.log_metrics({
                    'train_roc_auc': metrics['train_roc_auc'],
                    'val_roc_auc': metrics['val_roc_auc'],
                    'train_pr_auc': metrics['train_pr_auc'],
                    'val_pr_auc': metrics['val_pr_auc'],
                })

            # Log model
            mlflow.sklearn.log_model(model, "model")

        logger.info(f"Evaluated {model_name} - Val F1: {metrics['val_f1']:.4f}, Val ROC-AUC: {metrics.get('val_roc_auc', 0):.4f}")

        return metrics

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1'
    ) -> Dict:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dictionary with cross-validation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"Cross-validation {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        return cv_results

    def plot_confusion_matrix(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ):
        """
        Plot confusion matrix for a model.

        Args:
            model: Trained model
            X: Features
            y: True labels
            title: Plot title
            save_path: Path to save figure
        """
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pass', 'Fail'],
                    yticklabels=['Pass', 'Fail'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                       ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_roc_curve(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None
    ):
        """
        Plot ROC curve for a model.

        Args:
            model: Trained model
            X: Features
            y: True labels
            title: Plot title
            save_path: Path to save figure
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return

        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def plot_precision_recall_curve(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None
    ):
        """
        Plot precision-recall curve for a model.

        Args:
            model: Trained model
            X: Features
            y: True labels
            title: Plot title
            save_path: Path to save figure
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return

        y_proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
        ax.axhline(y=y.mean(), color='k', linestyle='--', linewidth=1, label=f'Baseline ({y.mean():.3f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def compare_models(
        self,
        save_path: Optional[Path] = None
    ):
        """
        Compare all evaluated models.

        Args:
            save_path: Path to save comparison figure
        """
        if not self.results:
            logger.warning("No models to compare")
            return

        results_df = pd.DataFrame(self.results)

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        metrics = ['val_f1', 'val_precision', 'val_recall', 'val_roc_auc']
        titles = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            if metric in results_df.columns:
                data = results_df.sort_values(metric, ascending=False)
                bars = ax.barh(data['model_name'], data[metric], color='steelblue', alpha=0.7)

                # Color best model
                bars[0].set_color('darkgreen')

                # Add values on bars
                for i, v in enumerate(data[metric]):
                    ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')

                ax.set_xlabel(title, fontsize=12)
                ax.set_title(f'Model Comparison - {title}', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3, axis='x')
                ax.set_xlim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        return results_df

    def get_classification_report(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> str:
        """
        Get detailed classification report.

        Args:
            model: Trained model
            X: Features
            y: True labels

        Returns:
            Classification report as string
        """
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, target_names=['Pass', 'Fail'])
        return report


def load_engineered_data(
    use_smote: bool = False,
    use_pca: bool = False,
    data_dir: str = "../data/engineered"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load engineered datasets.

    Args:
        use_smote: Whether to use SMOTE-balanced training data
        use_pca: Whether to use PCA-transformed data
        data_dir: Directory containing engineered data

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data_path = Path(data_dir)

    suffix = '_pca' if use_pca else ''
    train_file = 'train_engineered_smote.csv' if use_smote and not use_pca else f'train_engineered{suffix}.csv'

    # Load data
    train_data = pd.read_csv(data_path / train_file)
    val_data = pd.read_csv(data_path / f'val_engineered{suffix}.csv')
    test_data = pd.read_csv(data_path / f'test_engineered{suffix}.csv')

    # Split features and labels
    X_train = train_data.drop('Label', axis=1)
    y_train = train_data['Label']

    X_val = val_data.drop('Label', axis=1)
    y_val = val_data['Label']

    X_test = test_data.drop('Label', axis=1)
    y_test = test_data['Label']

    logger.info(f"Loaded data (SMOTE={use_smote}, PCA={use_pca})")
    logger.info(f"  Train: {X_train.shape}, Fail rate: {y_train.mean():.2%}")
    logger.info(f"  Val: {X_val.shape}, Fail rate: {y_val.mean():.2%}")
    logger.info(f"  Test: {X_test.shape}, Fail rate: {y_test.mean():.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None
):
    """
    Plot feature importance from tree-based models.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature importances")
        return

    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importances)), importances['Importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances['Feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
