"""Create dirty marketing data for week09 assignment."""

import numpy as np
import pandas as pd
from pathlib import Path

# Load original data
data_path = Path(__file__).parent.parent.parent.parent.parent / "homework" / "week06" / "data" / "q3_marketing.csv"
df = pd.read_csv(data_path, keep_default_na=False)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Add missing values (about 5% randomly)
n = len(df)
for col in ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']:
    mask = np.random.random(n) < 0.05
    df.loc[mask, col] = np.nan

# 2. Add extreme outliers (top 1% as very large values)
for col in ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']:
    outlier_mask = np.random.random(n) < 0.01
    df.loc[outlier_mask, col] = df[col].quantile(0.99) * np.random.uniform(2, 5, outlier_mask.sum())

# 3. Add more regions (expand Region column)
regions = ['EU', 'NA', 'Asia', 'LatAm']
df['Region'] = np.random.choice(regions, n)

# Save dirty data
output_path = Path(__file__).parent.parent / "data" / "dirty_marketing.csv"
df.to_csv(output_path, index=False)

print(f"Dirty data created: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nRegion distribution:\n{df['Region'].value_counts()}")
