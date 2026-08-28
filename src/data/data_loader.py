"""
Data Loader for UCI SECOM Semiconductor Manufacturing Dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SECOMDataLoader:
    """Load and preprocess the UCI SECOM dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.features = None
        self.labels = None
        self.timestamps = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load SECOM dataset from files (auto-detects CSV or separate .data files)."""
        logger.info("Loading SECOM dataset...")

        # Check for CSV format first
        csv_path = self.data_dir / "uci-secom.csv"
        if csv_path.exists():
            return self._load_from_csv(csv_path)

        # Fall back to separate .data files
        return self._load_from_data_files()

    def _load_from_csv(self, csv_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load SECOM dataset from CSV file."""
        logger.info(f"Loading from CSV: {csv_path}")

        # Load the CSV
        data = pd.read_csv(csv_path, na_values=['NaN', 'nan', '', 'NA'])

        # Extract timestamps (first column)
        self.timestamps = pd.to_datetime(data['Time'])

        # Extract labels (last column - Pass/Fail)
        # Convert: 1 = Fail, -1 = Pass -> 1 = Fail, 0 = Pass
        self.labels = (data['Pass/Fail'] == 1).astype(int)

        # Extract features (all columns except Time and Pass/Fail)
        feature_cols = [col for col in data.columns if col not in ['Time', 'Pass/Fail']]
        self.features = data[feature_cols].copy()

        # Rename columns to Feature_0, Feature_1, etc.
        self.features.columns = [f'Feature_{i}' for i in range(len(self.features.columns))]

        logger.info(f"Loaded {len(self.features)} observations with {self.features.shape[1]} features")
        logger.info(f"Class distribution: Pass={sum(self.labels==0)}, Fail={sum(self.labels==1)}")
        logger.info(f"Failure rate: {self.labels.mean():.2%}")

        return self.features, self.labels, self.timestamps

    def _load_from_data_files(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load SECOM dataset from separate .data files."""
        # Load features
        features_path = self.data_dir / "secom.data"
        if not features_path.exists():
            raise FileNotFoundError(
                f"No data files found in {self.data_dir}\n"
                "Please download SECOM dataset from: https://www.kaggle.com/datasets/paresh2047/uci-semcom\n"
                "Supported formats:\n"
                "  - uci-secom.csv (CSV format)\n"
                "  - secom.data + secom_labels.data (separate files)"
            )

        self.features = pd.read_csv(
            features_path,
            sep=r'\s+',
            header=None,
            na_values=['NaN', 'nan', '']
        )

        self.features.columns = [f'Feature_{i}' for i in range(self.features.shape[1])]

        # Load labels
        labels_path = self.data_dir / "secom_labels.data"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        labels_data = pd.read_csv(
            labels_path,
            sep=r'\s+',
            header=None,
            names=['Label', 'Timestamp']
        )

        self.labels = labels_data['Label']
        self.timestamps = labels_data['Timestamp']

        # Convert labels: -1 (pass) -> 0, 1 (fail) -> 1
        self.labels = (self.labels == 1).astype(int)

        logger.info(f"Loaded {len(self.features)} observations with {self.features.shape[1]} features")
        logger.info(f"Class distribution: Pass={sum(self.labels==0)}, Fail={sum(self.labels==1)}")
        logger.info(f"Failure rate: {self.labels.mean():.2%}")

        return self.features, self.labels, self.timestamps
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the dataset."""
        if self.features is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'n_samples': len(self.features),
            'n_features': self.features.shape[1],
            'n_pass': sum(self.labels == 0),
            'n_fail': sum(self.labels == 1),
            'failure_rate': self.labels.mean(),
            'missing_value_rate': self.features.isna().sum().sum() / (self.features.shape[0] * self.features.shape[1]),
            'features_with_missing': self.features.isna().any().sum(),
            'samples_with_missing': self.features.isna().any(axis=1).sum()
        }
        
        return summary
    
    def get_missing_value_analysis(self) -> pd.DataFrame:
        """Analyze missing values in the dataset."""
        if self.features is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_stats = pd.DataFrame({
            'Feature': self.features.columns,
            'Missing_Count': self.features.isna().sum(),
            'Missing_Percentage': (self.features.isna().sum() / len(self.features) * 100)
        })
        
        missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
        
        return missing_stats
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple:
        """Split data into train, validation, and test sets."""
        if self.features is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stratify_by = self.labels if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(self.features):.1%})")
        logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/len(self.features):.1%})")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(self.features):.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(
        self,
        output_dir: str = "data/processed",
        splits: Optional[Tuple] = None
    ):
        """Save processed data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if splits is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = splits
            
            train_data = X_train.copy()
            train_data['Label'] = y_train.values
            train_data.to_csv(output_path / "train.csv", index=False)
            
            val_data = X_val.copy()
            val_data['Label'] = y_val.values
            val_data.to_csv(output_path / "val.csv", index=False)
            
            test_data = X_test.copy()
            test_data['Label'] = y_test.values
            test_data.to_csv(output_path / "test.csv", index=False)
            
            logger.info(f"Saved train/val/test splits to {output_path}")
