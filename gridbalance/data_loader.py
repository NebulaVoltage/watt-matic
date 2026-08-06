"""
SGCC Data Loader & Schema Validator Module
Handles loading, column identification, and data structures for SGCC EDA.
"""

import os
import pandas as pd
import numpy as np
from generate_sample_data import generate_sgcc_sample

class SGCCDataLoader:
    def __init__(self, filepath="data.csv"):
        self.filepath = filepath
        self.df = None
        self.id_col = None
        self.target_col = None
        self.date_cols = []
        self.load_data()

    def load_data(self):
        """Loads dataset from filepath or generates synthetic dataset if missing."""
        if not os.path.exists(self.filepath):
            print(f"[INFO] File '{self.filepath}' not found. Generating sample SGCC dataset...")
            generate_sgcc_sample(filename=self.filepath)

        print(f"[INFO] Reading SGCC Dataset from '{self.filepath}'...")
        self.df = pd.read_csv(self.filepath)
        self._parse_schema()

    def _parse_schema(self):
        """Identifies CONS_NO, FLAG, and daily consumption date columns."""
        cols = self.df.columns.tolist()
        
        # Identify ID column
        if "CONS_NO" in cols:
            self.id_col = "CONS_NO"
        else:
            self.id_col = cols[0]

        # Identify Target column
        if "FLAG" in cols:
            self.target_col = "FLAG"
        elif "flag" in cols:
            self.target_col = "flag"
        else:
            self.target_col = cols[1]

        # Remaining columns are daily time series
        self.date_cols = [c for c in cols if c not in [self.id_col, self.target_col]]
        
        print(f"[SUCCESS] Dataset loaded successfully.")
        print(f" - Rows: {len(self.df):,}")
        print(f" - Columns: {len(cols):,}")
        print(f" - ID Column: {self.id_col}")
        print(f" - Target Column: {self.target_col}")
        print(f" - Daily Consumption Features: {len(self.date_cols):,}")

    def get_consumption_matrix(self, fill_na=None):
        """Returns only the numerical daily consumption columns."""
        mat = self.df[self.date_cols]
        if fill_na is not None:
            mat = mat.fillna(fill_na)
        return mat

    def get_normal_and_theft_subsets(self):
        """Returns separated DataFrames for Normal (0) and Theft (1) customers."""
        normal_df = self.df[self.df[self.target_col] == 0]
        theft_df = self.df[self.df[self.target_col] == 1]
        return normal_df, theft_df

if __name__ == "__main__":
    loader = SGCCDataLoader()
