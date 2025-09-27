"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset from given path."""
    df = pd.read_csv(path)
    return df


def default_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply some basic feature engineering."""
    df = df.copy()

    # Example: convert skills into count of skills
    if 'skills' in df.columns:
        df['skills_count'] = df['skills'].fillna("").apply(
            lambda s: 0 if s.strip() == "" else len([x for x in s.split(',') if x.strip()])
        )

    # Normalize job_title (lowercase)
    if 'job_title' in df.columns:
        df['job_title'] = df['job_title'].fillna('unknown').str.lower()

    # Handle missing education
    if 'education_level' in df.columns:
        df['education_level'] = df['education_level'].fillna('unknown')

    return df


def build_preprocessor(categorical_features, numeric_features):
    """Build preprocessing pipeline for numeric + categorical features."""

    # numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_test_split_features(df: pd.DataFrame,
                              target_col: str = 'salary',
                              test_size: float = 0.2,
                              random_state: int = 42):
    """Split dataset into train/test after feature engineering."""
    df = default_feature_engineering(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # quick test
    print('Run data_prep functions from train.py')
