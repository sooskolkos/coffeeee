"""
ETL Pipeline for data processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import click
from data_loader import load_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess data
    """
    df_clean = df.copy()
    
    print("Handling missing values...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            print(f"  - Column '{col}': {df_clean[col].isnull().sum()} missing values")
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    initial_shape = df_clean.shape
    df_clean = df_clean.drop_duplicates()
    final_shape = df_clean.shape
    removed_duplicates = initial_shape[0] - final_shape[0]
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features
    """
    df_featured = df.copy()
    
    print("Performing feature engineering...")
    
    date_columns = df_featured.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df_featured[col] = pd.to_datetime(df_featured[col], errors='ignore')
        except:
            pass
    
    date_columns = df_featured.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df_featured[f'{col}_year'] = df_featured[col].dt.year
        df_featured[f'{col}_month'] = df_featured[col].dt.month
        df_featured[f'{col}_day'] = df_featured[col].dt.day
        print(f"  - Created date features for '{col}'")
    
    return df_featured


@click.command()
@click.option('--output', default='data/processed/cleaned_data.csv', help='Output data path')
@click.option('--local-file', default=None, help='Optional local file path to load instead of Google Drive')
def run_etl_pipeline(output: str, local_file: str):
    """
    Run ETL pipeline
    """
    print("Starting ETL pipeline...")
    
    print("Step 1: Extracting data...")
    df = load_data(local_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("Step 2: Transforming data...")
    df_clean = clean_data(df)
    df_final = feature_engineering(df_clean)
    
    print("Step 3: Loading data...")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output, index=False)
    
    print(f"ETL pipeline completed!")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Processed data saved to: {output}")
    
    print("\nFirst 5 rows of processed data:")
    print(df_final.head())


if __name__ == "__main__":
    run_etl_pipeline()