import argparse
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def main():
    parser = argparse.ArgumentParser(description="Week09 Data Cleaner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.fillna(df.mean(numeric_only=True))

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    df.to_csv(args.output, index=False)
    print("✅ Clean data saved.")

if __name__ == "__main__":
    main()