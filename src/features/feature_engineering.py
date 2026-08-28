"""
Feature Engineering Pipeline for SECOM Semiconductor Dataset.

This module provides classes and functions for preprocessing and engineering
features from the SECOM dataset, including:
- Missing value imputation
- Feature scaling and transformation
- Feature selection
- Class imbalance handling
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging

logger = logging.getLogger(__name__)


class SECOMFeatureEngineer:
    """Feature engineering pipeline for SECOM dataset."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the feature engineer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.removed_features = []
        self.imputers = {}
        self.scalers = {}
        self.feature_selector = None
        self.pca = None
        self.resampler = None

    def remove_high_missing_features(
        self,
        X: pd.DataFrame,
        threshold: float = 40.0
    ) -> pd.DataFrame:
        """
        Remove features with missing values above threshold.

        Args:
            X: Feature dataframe
            threshold: Missing percentage threshold (0-100)

        Returns:
            DataFrame with high-missing features removed
        """
        missing_pct = (X.isna().sum() / len(X) * 100)
        high_missing = missing_pct[missing_pct > threshold].index.tolist()

        logger.info(f"Removing {len(high_missing)} features with >{threshold}% missing values")
        self.removed_features.extend(high_missing)

        return X.drop(columns=high_missing)

    def remove_zero_variance_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Remove features with zero or near-zero variance.

        Args:
            X: Feature dataframe
            threshold: Variance threshold

        Returns:
            DataFrame with zero-variance features removed
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X.fillna(X.median()))

        zero_var_features = X.columns[~selector.get_support()].tolist()
        logger.info(f"Removing {len(zero_var_features)} zero-variance features")

        self.removed_features.extend(zero_var_features)
        return X.drop(columns=zero_var_features)

    def impute_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'auto'
    ) -> pd.DataFrame:
        """
        Impute missing values using different strategies based on missing rate.

        Args:
            X: Feature dataframe
            strategy: 'auto', 'median', 'knn', or 'iterative'

        Returns:
            DataFrame with imputed values
        """
        X_imputed = X.copy()

        if strategy == 'auto':
            # Categorize features by missing percentage
            missing_pct = (X.isna().sum() / len(X) * 100)

            # Low missing (<5%): Simple median imputation
            low_missing = missing_pct[(missing_pct > 0) & (missing_pct < 5)].index.tolist()
            if low_missing:
                logger.info(f"Simple imputation for {len(low_missing)} features with <5% missing")
                imputer = SimpleImputer(strategy='median')
                X_imputed[low_missing] = imputer.fit_transform(X[low_missing])
                self.imputers['simple'] = imputer

            # Medium missing (5-20%): KNN imputation
            medium_missing = missing_pct[(missing_pct >= 5) & (missing_pct <= 20)].index.tolist()
            if medium_missing:
                logger.info(f"KNN imputation for {len(medium_missing)} features with 5-20% missing")
                imputer = KNNImputer(n_neighbors=5)
                X_imputed[medium_missing] = imputer.fit_transform(X[medium_missing])
                self.imputers['knn'] = imputer

            # High missing (20-40%): Iterative imputation
            high_missing = missing_pct[(missing_pct > 20) & (missing_pct <= 40)].index.tolist()
            if high_missing:
                logger.info(f"Iterative imputation for {len(high_missing)} features with 20-40% missing")
                imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
                X_imputed[high_missing] = imputer.fit_transform(X[high_missing])
                self.imputers['iterative'] = imputer

        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.imputers['median'] = imputer

        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.imputers['knn'] = imputer

        elif strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.imputers['iterative'] = imputer

        logger.info("Missing value imputation completed")
        return X_imputed

    def transform_skewed_features(
        self,
        X: pd.DataFrame,
        skew_threshold: float = 2.0,
        method: str = 'yeo-johnson'
    ) -> pd.DataFrame:
        """
        Apply power transformation to highly skewed features.

        Args:
            X: Feature dataframe
            skew_threshold: Absolute skewness threshold for transformation
            method: 'yeo-johnson' or 'box-cox'

        Returns:
            DataFrame with transformed features
        """
        X_transformed = X.copy()
        skewness = X.skew()
        skewed_features = skewness[abs(skewness) > skew_threshold].index.tolist()

        if skewed_features:
            logger.info(f"Transforming {len(skewed_features)} highly skewed features")
            transformer = PowerTransformer(method=method)
            X_transformed[skewed_features] = transformer.fit_transform(X[skewed_features])
            self.scalers['power_transformer'] = transformer

        return X_transformed

    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Scale features using specified method.

        Args:
            X: Feature dataframe
            method: 'standard' or 'robust'

        Returns:
            Scaled feature dataframe
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        self.scalers[method] = scaler
        logger.info(f"Features scaled using {method} scaling")

        return X_scaled

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'kbest',
        k: int = 100
    ) -> pd.DataFrame:
        """
        Select top k features based on statistical tests.

        Args:
            X: Feature dataframe
            y: Target labels
            method: Feature selection method
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)

            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(
                X_selected,
                columns=selected_features,
                index=X.index
            )

            self.feature_selector = selector
            logger.info(f"Selected top {len(selected_features)} features using {method}")

            return X_selected

        return X

    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: Optional[float] = 0.95
    ) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply PCA for dimensionality reduction.

        Args:
            X: Feature dataframe
            n_components: Number of components or variance to preserve

        Returns:
            Tuple of (transformed dataframe, PCA object)
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        n_comp = X_pca.shape[1]
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC_{i+1}' for i in range(n_comp)],
            index=X.index
        )

        self.pca = pca
        logger.info(f"PCA applied: {n_comp} components preserve {n_components*100:.1f}% variance")

        return X_pca_df, pca

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'smote',
        sampling_strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using oversampling techniques.

        Args:
            X: Feature dataframe
            y: Target labels
            method: 'smote', 'adasyn', 'smote_tomek', or 'undersample'
            sampling_strategy: Sampling strategy

        Returns:
            Tuple of (resampled features, resampled labels)
        """
        if method == 'smote':
            resampler = SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
        elif method == 'adasyn':
            resampler = ADASYN(random_state=self.random_state, sampling_strategy=sampling_strategy)
        elif method == 'smote_tomek':
            resampler = SMOTETomek(random_state=self.random_state, sampling_strategy=sampling_strategy)
        elif method == 'undersample':
            resampler = RandomUnderSampler(random_state=self.random_state, sampling_strategy=sampling_strategy)
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        X_resampled, y_resampled = resampler.fit_resample(X, y)

        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        self.resampler = resampler

        original_dist = y.value_counts().to_dict()
        new_dist = y_resampled.value_counts().to_dict()

        logger.info(f"Class imbalance handled using {method}")
        logger.info(f"  Original: {original_dist}")
        logger.info(f"  Resampled: {new_dist}")

        return X_resampled, y_resampled

    def create_interaction_features(
        self,
        X: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features from specified feature pairs.

        Args:
            X: Feature dataframe
            feature_pairs: List of (feature1, feature2) tuples

        Returns:
            DataFrame with added interaction features
        """
        X_with_interactions = X.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                X_with_interactions[interaction_name] = X[feat1] * X[feat2]

        logger.info(f"Created {len(feature_pairs)} interaction features")
        return X_with_interactions

    def create_polynomial_features(
        self,
        X: pd.DataFrame,
        features: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for specified features.

        Args:
            X: Feature dataframe
            features: List of feature names
            degree: Polynomial degree

        Returns:
            DataFrame with added polynomial features
        """
        X_with_poly = X.copy()

        for feat in features:
            if feat in X.columns:
                for d in range(2, degree + 1):
                    poly_name = f'{feat}_pow{d}'
                    X_with_poly[poly_name] = X[feat] ** d

        logger.info(f"Created polynomial features (degree {degree}) for {len(features)} features")
        return X_with_poly

    def get_feature_importance_summary(self) -> Dict:
        """
        Get summary of feature engineering steps and selected features.

        Returns:
            Dictionary with feature engineering summary
        """
        summary = {
            'removed_features_count': len(self.removed_features),
            'removed_features': self.removed_features,
            'imputation_methods': list(self.imputers.keys()),
            'scaling_methods': list(self.scalers.keys()),
            'feature_selector_used': self.feature_selector is not None,
            'pca_used': self.pca is not None,
            'resampler_used': self.resampler is not None
        }

        if self.pca is not None:
            summary['pca_n_components'] = self.pca.n_components_
            summary['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())

        if self.feature_selector is not None:
            summary['n_features_selected'] = self.feature_selector.get_support().sum()

        return summary


def create_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    use_smote: bool = True,
    use_pca: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, SECOMFeatureEngineer]:
    """
    Create full feature engineering pipeline.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        X_test: Test features
        use_smote: Whether to apply SMOTE
        use_pca: Whether to apply PCA
        random_state: Random seed

    Returns:
        Tuple of (processed X_train, y_train, X_val, X_test, engineer object)
    """
    engineer = SECOMFeatureEngineer(random_state=random_state)

    # Step 1: Remove high missing features (>40%)
    X_train = engineer.remove_high_missing_features(X_train, threshold=40.0)
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    # Step 2: Remove zero variance features
    X_train = engineer.remove_zero_variance_features(X_train, threshold=0.0)
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    # Step 3: Impute missing values
    X_train = engineer.impute_missing_values(X_train, strategy='auto')
    X_val = engineer.impute_missing_values(X_val, strategy='auto')
    X_test = engineer.impute_missing_values(X_test, strategy='auto')

    # Step 4: Transform skewed features
    X_train = engineer.transform_skewed_features(X_train, skew_threshold=2.0)
    X_val = engineer.transform_skewed_features(X_val, skew_threshold=2.0)
    X_test = engineer.transform_skewed_features(X_test, skew_threshold=2.0)

    # Step 5: Scale features
    X_train = engineer.scale_features(X_train, method='standard')
    X_val = engineer.scale_features(X_val, method='standard')
    X_test = engineer.scale_features(X_test, method='standard')

    # Step 6: Apply PCA if requested
    if use_pca:
        X_train, _ = engineer.apply_pca(X_train, n_components=0.95)
        X_val, _ = engineer.apply_pca(X_val, n_components=0.95)
        X_test, _ = engineer.apply_pca(X_test, n_components=0.95)

    # Step 7: Handle class imbalance (only on training set)
    if use_smote:
        X_train, y_train = engineer.handle_class_imbalance(X_train, y_train, method='smote')

    return X_train, y_train, X_val, X_test, engineer
