"""
Script to load and perform initial exploration of SECOM dataset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import SECOMDataLoader
from src.utils.helpers import setup_logging

logger = setup_logging()


def main():
    """Main function to load and explore SECOM data."""
    
    print("="*80)
    print("SECOM DATASET LOADING AND INITIAL EXPLORATION")
    print("="*80)
    
    # Initialize loader
    loader = SECOMDataLoader("data/raw")
    
    try:
        # Load data
        features, labels, timestamps = loader.load_data()
        
        # Get summary
        summary = loader.get_data_summary()
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total Samples:              {summary['n_samples']:,}")
        print(f"Total Features:             {summary['n_features']:,}")
        print(f"Pass (Class 0):             {summary['n_pass']:,} ({summary['n_pass']/summary['n_samples']:.1%})")
        print(f"Fail (Class 1):             {summary['n_fail']:,} ({summary['n_fail']/summary['n_samples']:.1%})")
        print(f"Failure Rate:               {summary['failure_rate']:.2%}")
        print(f"Overall Missing Rate:       {summary['missing_value_rate']:.2%}")
        print(f"Features with Missing:      {summary['features_with_missing']:,}")
        print(f"Samples with Missing:       {summary['samples_with_missing']:,}")
        
        # Class imbalance ratio
        imbalance_ratio = summary['n_pass'] / summary['n_fail']
        print(f"\nClass Imbalance Ratio:      1:{imbalance_ratio:.1f} (Pass:Fail)")
        
        # Missing value analysis
        missing_analysis = loader.get_missing_value_analysis()
        
        print("\n" + "="*80)
        print("TOP 20 FEATURES WITH MOST MISSING VALUES")
        print("="*80)
        print(missing_analysis.head(20).to_string(index=False))
        
        # Features with high missing rate
        high_missing = missing_analysis[missing_analysis['Missing_Percentage'] > 40]
        print(f"\n{len(high_missing)} features have >40% missing values")
        print("These will likely be removed during preprocessing")
        
        # Basic statistics
        print("\n" + "="*80)
        print("SAMPLE FEATURE STATISTICS (First 5 features)")
        print("="*80)
        print(features.iloc[:, :5].describe())
        
        # Create train/test split
        print("\n" + "="*80)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("="*80)
        
        splits = loader.train_test_split(
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
            stratify=True
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        print(f"\nTrain set - Fail rate: {y_train.mean():.2%}")
        print(f"Val set   - Fail rate: {y_val.mean():.2%}")
        print(f"Test set  - Fail rate: {y_test.mean():.2%}")
        
        # Save processed data
        print("\n" + "="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)
        
        loader.save_processed_data(splits=splits)
        print("✓ Saved train.csv, val.csv, test.csv to data/processed/")
        
        # Quick visualization
        print("\n" + "="*80)
        print("CREATING QUICK VISUALIZATIONS")
        print("="*80)
        
        # Create reports directory
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        
        # 1. Class distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar plot
        class_counts = labels.value_counts()
        axes[0].bar(['Pass', 'Fail'], [class_counts[0], class_counts[1]], 
                    color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for i, v in enumerate([class_counts[0], class_counts[1]]):
            axes[0].text(i, v, f'{v}\n({v/len(labels):.1%})', 
                        ha='center', va='bottom')
        
        # Pie chart
        axes[1].pie([class_counts[0], class_counts[1]], 
                    labels=['Pass', 'Fail'],
                    autopct='%1.1f%%',
                    colors=['green', 'red'],
                    alpha=0.7)
        axes[1].set_title('Class Proportion')
        
        plt.tight_layout()
        plt.savefig('reports/figures/01_class_distribution.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: reports/figures/01_class_distribution.png")
        plt.close()
        
        # 2. Missing values distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        missing_pct = missing_analysis['Missing_Percentage'].values
        ax.hist(missing_pct, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(40, color='red', linestyle='--', label='40% threshold')
        ax.set_xlabel('Missing Percentage')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Missing Values Across Features')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/02_missing_values_distribution.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: reports/figures/02_missing_values_distribution.png")
        plt.close()
        
        # 3. Sample feature distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(6):
            feature_col = features.columns[i]
            data = features[feature_col].dropna()
            
            axes[i].hist(data, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{feature_col}\n(Mean: {data.mean():.2f})')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/03_sample_feature_distributions.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: reports/figures/03_sample_feature_distributions.png")
        plt.close()
        
        print("\n" + "="*80)
        print("DATA LOADING COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Open notebooks/01_data_exploration.ipynb for detailed EDA")
        print("2. Review visualizations in reports/figures/")
        print("3. Start feature engineering with notebooks/02_feature_engineering.ipynb")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS FOR MODELING:")
        print("="*80)
        print(f"• Highly imbalanced dataset (1:{imbalance_ratio:.0f}) - Use SMOTE or class weights")
        print(f"• {summary['features_with_missing']} features have missing values - Need imputation")
        print(f"• {len(high_missing)} features have >40% missing - Consider removing")
        print(f"• 590 features - High dimensional, need feature selection")
        print(f"• Only {summary['n_fail']} failure samples - Limited positive examples")
        
    except FileNotFoundError as e:
        logger.error(f"\n{str(e)}")
        logger.error("\nPlease ensure secom.data and secom_labels.data are in data/raw/")


if __name__ == "__main__":
    main()
