"""
- Rename and clean columns
- Building a profile_text (biography) from many columns
- categorical encoding
- normalize scaling
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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
        self.profile_text       = profile_text
        self.categorical_fields = categorical_fields
        self.numeric_fields     = numeric_fields

        # ColumnTransformer Fit
        self.column_transformer: Optional[ColumnTransformer] = None
        self.fitted = False
    
    @staticmethod
    def rename_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename all columns in-place for the given column name map in config.py
        """
        df = df.rename(columns=config.RENAME_MAP)
        
        # lower all columns 
        df.columns = [col.lower() for col in df.columns]
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
        df_copy = df.copy()
        
        # Standardize all columns to str
        for col in df_copy.columns:
            # Fill empty fields with empty string + str cast
            df_copy[col] = df_copy[col].fillna("").astype(str)
        
        # Numeric coercion for numeric fields
        for col in self.numeric_fields:
            if col in df_copy.columns:
                # Cast integer if possible, else NaN upon TypeError
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce', downcast='float')
                df_copy[col] = df_copy[col].fillna(0).astype(float)
        return df_copy

    def fit(self, df: pd.DataFrame):
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
        print("Fitting DataFrame...")

        df = self.rename_column(df)
        df = self._clean_table(df)
        df['profile_text'] = self.build_profile_text(df, self.profile_text)
        
        # Building our ColumnTransformer through Onehot & StandardScaler
        transformers = []

        ##categorical_fields : One Hot Encoidng
        available_category = [c for c in self.categorical_fields if c in df.columns]
        if available_category:
            ohe = OneHotEncoder(handle_unknown='error', sparse_output=False)
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
        x_meta = self.column_transformer.fit_transform(df)
        self.fitted = True
        print("Finished Fitting.")
        print("Fit Status: ", self.fitted)
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        print("Transforming...")

        if not self.fitted or self.column_transformer is None:
            raise RuntimeError("FeatureEngineer must be fitted before transform(). Call fit() first.")

        df = self.rename_column(df)
        df = self._clean_table(df)
        df['profile_text'] = self.build_profile_text(df, self.profile_text)
        
        print("Our current DataFrame", df.head())

        x_meta = self.column_transformer.transform(df)
        # If OneHotEncoder is Sparse -> cast to dense | Empty fields are filled with zeros
        if hasattr(x_meta, "toarray"):
            x_meta = x_meta.toarray()
        
        print("Finished Transforming.")
        return {
            'profile_text': df['profile_text'].to_numpy(),
            'meta_features': x_meta,
            'index': df.index.to_numpy(),
            'raw_df': df
        }
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.fit(df)
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
        extro_complement = 1.0 - np.abs(df['extroversion'] - 0.5)
        diversity_features.append(extro_complement)

        # Study Social Balance
        study_social_balance = np.abs(df['study_frequency'] - df['gym+frequency'])
        diversity_features.append(study_social_balance)

        return np.column_stack(diversity_features)