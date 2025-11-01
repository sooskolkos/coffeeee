"""
Data loader script for loading dataset directly from Google Drive
"""

import pandas as pd
from pathlib import Path
import os


def load_data_from_drive() -> pd.DataFrame:
    """
    Load dataset directly from Google Drive
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    FILE_ID = "1vpSvMbFClBkrYE7MaO7Z2OXYjDnFqEjm"
    file_url = f"https://drive.google.com/uc?id={FILE_ID}"
    
    try:
        print("Loading dataset from Google Drive...")
        raw_data = pd.read_csv(file_url)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {raw_data.shape}")
        return raw_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def save_data_locally(df: pd.DataFrame, file_path: str = "data/raw/dataset.csv") -> str:
    """
    Save dataset locally for backup
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the file
        
    Returns:
        str: Path to saved file
    """

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    

    df.to_csv(file_path, index=False)
    print(f"Dataset saved locally to: {file_path}")
    return file_path


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Main function to load data - tries local file first, then Google Drive
    
    Args:
        file_path: Optional local file path
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading dataset from local file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        return df
    
    df = load_data_from_drive()
    
    if file_path:
        save_data_locally(df, file_path)
    
    return df


if __name__ == "__main__":
    df = load_data()
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nDataset description:")
    print(df.describe())