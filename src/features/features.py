"""
- Rename and clean columns
- Building a profile_text (biography) from many columns
- categorical encoding
- normalize scaling
"""

import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


import config

class FeatureEngineer:
    def __init__(
            self,
            profile_text: Optional[List[str]]       = None,
            categorical_fields: Optional[list[str]] = None,
            numeric_fields: Optional[List[str]]     = None
    ):
        self.profile_text      : List[str] = profile_text
        self.categorical_fields: List[str] = categorical_fields
        self.numeric_fields    : List[str] = numeric_fields
        self.rename_map        : dict = None
        # ColumnTransformer Fit
        self.column_transformer: Optional[ColumnTransformer] = None
        self.fitted: bool  = False

        # Output
        self.x_meta: pd.DataFrame = None
        self.df: pd.DataFrame     = None
    
    @staticmethod
    def rename_column(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
        """
        Rename all columns in-place for the given column name map in config.py
        """
        df = df.rename(columns = rename_map)


        # Perform string casting on all column names, remove trailing space, & lower all names
        # List comprehension handling None values - when users did fill out Rename_Map right
        df.columns = [
            col if col is None else str(col).strip().lower()
            for col in df.columns
        ]
        """ Same purpose but much lengthier
        new_cols = []
        for col in df.columns:
            if col is not None:
                new_cols.append(str(col).strip().lower())
            else:
                new_cols.append(col)
        df.columns = new_cols
        """
        
        # Check if there are duplicat column names
        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names after lowercasing")

        return df
    
    @staticmethod
    def build_profile_text(df: pd.DataFrame, text_fields: List[str]) -> pd.Series:
        """
        Combine our profile_text fields together by checking cases for each row

        Return:
            A 1D array or pd.Series that includes our profile_text
        """
        def join_row(row) -> str:
            """
            Check each row for:
                1. None values
                2. NaN values 
                3. String Casting
            Return: A Str
            """
            parts = []
            for field in text_fields:
                # verify the field exist, else it's empty str
                candidate_text = row.get(field, "")
                # If NaN -> empty str
                if pd.isna(candidate_text):
                    candidate_text = ""

                candidate_text = str(candidate_text).strip()
                if candidate_text:
                    parts.append(candidate_text)

            # If multiple text field, delimit by a period
            if parts:
                return ". ".join(parts) 
            else:
                return ""
        
        return (df.apply(join_row, axis=1)) # Axis=1 : apply for each row at O(n) times

    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove NaN, normalize whitespace/trailing spaces & consistent string types

        Return:
            A new copy of cleaned DataFrame
        """
        # Shallow Copy?
        df = df.copy()
        
        # Standardize categorical fields
        for col in self.categorical_fields:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna(np.nan)  # For OntHotEncoding to handle NaN values, where it uses np.isnan()
                    .astype(str)
                    .str.strip()
                )

        # Standardize Numerical Fields
        for col in self.numeric_fields:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    def fit(self, df: pd.DataFrame,  rename_map: Optional[dict]):
        """
        Performing preprocessing transformation on profile_text, categorical_fields, numeric_fields
        where we transformed:
            profile_text: : List[str]
            categorical_fields: One Hot Encoding
            numeric_field: Standard Scaler 
        with the following steps:
        1. Standardize all columns w/ rename
        2. Clean all entries w/ trailing white space, string types, and NaN values
        3. Clean profile text
        
        Then, perform:
        1. OneHotEncoding
        2. Standard Scaler
        3. Column Transformer
        Output: 
            update column_transforemr with our given DataFrame with a single matrix
        """
        self.rename_map = rename_map 
        if self.rename_map:
            df = self.rename_column(df, self.rename_map)

        df = self._clean_table(df)
        
        # Building our ColumnTransformer through Onehot & StandardScaler
        transformers = []

        ##categorical_fields : One Hot Encoidng
        available_category = [c for c in self.categorical_fields if c in df.columns]
        if available_category:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            transformers.append(("OneHot" , ohe, available_category))

        ## numeric_fields : StandardScaler | Mean=0, Var=1 
        available_nums = [c for c in self.numeric_fields if c in df.columns]
        if available_nums:
            scaler = StandardScaler()
            transformers.append(('likert_scale', scaler, available_nums))
            
        if not transformers:
            raise ValueError("No categorical or numeric features available to fit ColumnTransformer.")

        self.column_transformer = ColumnTransformer(transformers=transformers, 
                                                    remainder='drop') # Drop other features not mentioned
        
        # Fit using ColumnTransformer
        self.column_transformer.fit(df)
        self.fitted = True
        print(f"Fitted ColumnTransformer on {len(df)} samples")
        print(f"  - Categorical features: {available_category}")
        print(f"  - Numerical features: {available_nums}")
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        print("Transforming...")

        if not self.fitted or self.column_transformer is None:
            raise RuntimeError("Features must be fitted before transform(). Call fit() first.")
        
        if self.rename_map:
            df = self.rename_column(df, self.rename_map)

        df = self._clean_table(df)

        df['profile_text'] = self.build_profile_text(df, self.profile_text)
        
        # Transform using ColumnTransformer
        x_meta = self.column_transformer.transform(df)

        # If metadata is sparse, we convert to dense
        if hasattr(x_meta, "to_array"):
            x_meta = x_meta.to_array()

        print(f"Transformed features shape: {x_meta.shape}")

        assert x_meta.shape[0] == len(df), \
            f"Mismatch: x_meta has {x_meta.shape[0]} samples but df has {len(df)}"


        return {
            'profile_text': df['profile_text'].to_numpy(),
            'meta_features': x_meta,
            'index': df.index.to_numpy(),
            'raw_df': df
        }
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.fit(df, self.rename_map)
        return self.transform(df)

    def compute_diversity_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features that helps identify complementary pairings
        Example: extroversion difference, major overlap, hobby diversity

        Goal:
        1. Balancing Extroversion, so that we do not an extremely introverted & extroverted pairings
        2. Balance Study and Social
        """
        diversity_features = []

        # Extroversion complementary 
        extro_complement = 1.0 - np.abs((df['extroversion'] - 3) / 2)
        diversity_features.append(extro_complement)

        # Study Social Balance
        study_social_balance = np.abs(df['study_frequency'] - df['gym_frequency'])
        diversity_features.append(study_social_balance)

        return np.column_stack(diversity_features)