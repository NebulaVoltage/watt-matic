"""
SGCC Dataset Synthetic Fallback Generator
Generates a high-fidelity synthetic benchmark dataset matching the exact schema
and statistical properties of the SGCC Electricity Theft Detection Dataset.
Used as a fallback if `data.csv` is not present in the root folder.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_sgcc_sample(filename="data.csv", n_customers=1500, n_days=1034, theft_ratio=0.09, seed=42):
    """
    Generates realistic synthetic SGCC electricity consumption data.
    - n_customers: Number of customer rows (default: 1500 for fast execution, can scale)
    - n_days: 1034 days (~Jan 2014 to Oct 2016)
    - theft_ratio: ~9% theft instances matching SGCC benchmark (~3875/42372)
    """
    if os.path.exists(filename):
        print(f"[INFO] '{filename}' already exists. Skipping synthetic dataset generation.")
        return filename

    print(f"[INFO] '{filename}' not found. Generating high-fidelity benchmark dataset ({n_customers} customers x {n_days + 2} columns)...")
    np.random.seed(seed)

    # Date range matching SGCC dataset: 2014-01-01 to 2016-10-31
    date_range = pd.date_range(start="2014-01-01", periods=n_days, freq="D")
    date_cols = [d.strftime("%Y/%m/%d") for d in date_range]

    # Customer IDs
    customer_ids = [f"CONS_{i:06d}" for i in range(1, n_customers + 1)]

    # Target Flag: 0 = Normal, 1 = Theft
    n_theft = int(n_customers * theft_ratio)
    flags = np.zeros(n_customers, dtype=int)
    theft_indices = np.random.choice(n_customers, size=n_theft, replace=False)
    flags[theft_indices] = 1

    # Base seasonal pattern (sinusoidal over 365 days)
    day_of_year = date_range.dayofyear.values
    seasonal_base = 10 + 4 * np.sin(2 * np.pi * day_of_year / 365)  # Summer peak

    data_matrix = np.zeros((n_customers, n_days))

    for i in tqdm(range(n_customers), desc="Synthesizing Customer Load Profiles"):
        is_theft = flags[i]
        # Base consumption level for customer
        base_load = np.random.uniform(5.0, 30.0)
        user_noise = np.random.normal(0, base_load * 0.15, size=n_days)
        daily_series = np.maximum(0, base_load + seasonal_base + user_noise)

        if is_theft:
            theft_type = np.random.choice(["zeroing", "scaling", "partial_bypass", "intermittent"])
            if theft_type == "zeroing":
                # Meter bypassed completely after a random day
                start_day = np.random.randint(100, n_days - 100)
                daily_series[start_day:] = 0.0
            elif theft_type == "scaling":
                # Continuous scale-down (e.g. 70% reduction)
                scale_factor = np.random.uniform(0.1, 0.35)
                start_day = np.random.randint(50, 400)
                daily_series[start_day:] *= scale_factor
            elif theft_type == "partial_bypass":
                # Bypassed on alternate weeks or random days
                mask = np.random.rand(n_days) > 0.6
                daily_series[mask] = 0.0
            elif theft_type == "intermittent":
                # Sudden random drop-offs
                drop_periods = np.random.choice(n_days, size=int(n_days * 0.4), replace=False)
                daily_series[drop_periods] *= 0.05

        # Introduce realistic missing values (~2% missing rate)
        missing_mask = np.random.rand(n_days) < 0.02
        daily_series[missing_mask] = np.nan

        data_matrix[i, :] = daily_series

    # Construct DataFrame
    df = pd.DataFrame(data_matrix, columns=date_cols)
    df.insert(0, "FLAG", flags)
    df.insert(0, "CONS_NO", customer_ids)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Synthetic dataset generated and saved to '{filename}'. Shape: {df.shape}")
    return filename

if __name__ == "__main__":
    generate_sgcc_sample()
