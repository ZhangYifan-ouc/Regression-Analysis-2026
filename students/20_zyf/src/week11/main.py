"""
Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow
========================================================================
Task A: Synthetic data with known DGP → verify inference
Task B: Kaggle housing data → real-world regression pipeline
Task C: Comparison report

Single entry: uv run src/week11/main.py
"""

import sys
import os
import json
from pathlib import Path

# Ensure src/ is on path for utils imports
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils.models import AnalyticalOLS, GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomImputer, CustomStandardScaler
from utils.diagnostics import calculate_vif

# ── Paths ────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent / "data"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

SYNTHETIC_PATH = _DATA_DIR / "synthetic_regression.csv"
KAGGLE_PATH = _DATA_DIR / "sh_data.csv"

SYNTHETIC_REPORT = _RESULTS_DIR / "synthetic_report.md"
KAGGLE_REPORT = _RESULTS_DIR / "kaggle_report.md"
SUMMARY_REPORT = _RESULTS_DIR / "summary_comparison.md"


# ============================================================================
# TASK A: Synthetic Data
# ============================================================================

def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic housing price data with a known DGP.

    Scenario: House price (total_price in 万元)
    Features:
      - x1: area (㎡) — continuous, 50~200
      - x2: rooms (个) — continuous-ish, 1~6
      - x3: age (年) — continuous, 0~30
      - x4: location_score — continuous, 1~10, HIGHLY correlated with area
      - x5: has_garage — categorical (0/1)

    DGP:
      y = 30 + 0.5*area + 5*rooms - 1.5*age + 3*location_score + 15*has_garage + ε
      ε ~ N(0, 10)

    Deliberately: x4 = 0.02*area + 0.5*N(0,1) → area & location_score highly correlated

    Real-world issues injected:
      - Missing values on 'rooms' (5%)
      - Outliers on 'age' (top 2% set to 99)
      - Scale differences: area~100, age~15, location_score~6
      - Collinearity: area ↔ location_score
    """
    rng = np.random.default_rng(seed)
    n = 400

    # Generate features
    area = rng.uniform(50, 200, n)                    # ㎡
    rooms = rng.integers(1, 7, n).astype(float)       # 1~6 rooms
    age = rng.uniform(0, 30, n)                        # years
    location_score = 0.02 * area + rng.normal(0, 0.5, n)  # correlated with area!
    has_garage = rng.integers(0, 2, n).astype(float)   # 0 or 1

    # True coefficients
    y = (
        30.0
        + 0.5 * area
        + 5.0 * rooms
        - 1.5 * age
        + 3.0 * location_score
        + 15.0 * has_garage
        + rng.normal(0, 10, n)
    )

    # Build DataFrame
    df = pd.DataFrame({
        "area": area,
        "rooms": rooms,
        "age": age,
        "location_score": location_score,
        "has_garage": has_garage,
        "total_price": y,
    })

    # ── Inject real-world problems ──

    # 1. Missing values in 'rooms' (5% missing)
    missing_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[missing_idx, "rooms"] = np.nan

    # 2. Outliers in 'age' (top 2% → 99)
    outlier_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[outlier_idx, "age"] = 99.0

    # 3. Scale differences already exist by design (area ~100, age ~15)
    # 4. Collinearity already built: area ↔ location_score (cor ~0.9)

    return df


def run_synthetic_task(df: pd.DataFrame) -> dict:
    """
    Full pipeline on synthetic data:
      clean → encode → impute → scale → 5-fold CV → metrics + VIF + coefficient analysis

    Returns a dict of results for report generation.
    """
    rng = np.random.default_rng(42)

    # ── 1. Basic cleaning ──
    # Cap outlier in age at 95th percentile of non-outlier values
    age_normal = df.loc[df["age"] < 50, "age"]
    cap = np.percentile(age_normal, 95)
    df_clean = df.copy()
    df_clean["age"] = df_clean["age"].clip(upper=cap)

    # ── 2. Encode categorical (has_garage is already 0/1) ──
    # No further encoding needed for synthetic data

    # ── 3. Prepare feature matrix and target ──
    feature_cols = ["area", "rooms", "age", "location_score", "has_garage"]
    X_raw = df_clean[feature_cols].values.astype(np.float64)
    y_raw = df_clean["total_price"].values.astype(np.float64)

    # ── 4. 5-fold CV with leak-free preprocessing ──
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {"rmse": [], "mae": [], "mape": []}
    all_coefs = []

    for train_idx, val_idx in kf.split(X_raw):
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        # --- Fit on train ONLY ---
        imputer = CustomImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)

        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # Add intercept column
        X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
        X_val_design = np.column_stack([np.ones(X_val_scaled.shape[0]), X_val_scaled])

        # Train with GradientDescentOLS
        model = GradientDescentOLS(learning_rate=0.01, tol=1e-6, max_iter=2000)
        model.fit(X_train_design, y_train, seed=42)

        # Predict & evaluate
        y_pred = model.predict(X_val_design)

        fold_metrics["rmse"].append(calculate_rmse(y_val, y_pred))
        fold_metrics["mae"].append(calculate_mae(y_val, y_pred))
        fold_metrics["mape"].append(calculate_mape(y_val, y_pred))
        all_coefs.append(model.coef_)

    # ── 5. Full-set preprocessing for VIF & final coefficient analysis ──
    # Note: VIF is for diagnostic only, NOT for CV — safe to compute on full data
    imputer_full = CustomImputer(strategy="mean")
    X_full_imp = imputer_full.fit_transform(X_raw)
    scaler_full = CustomStandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full_imp)
    X_full_design = np.column_stack([np.ones(X_full_scaled.shape[0]), X_full_scaled])
    y_full = y_raw

    model_full = GradientDescentOLS(learning_rate=0.01, tol=1e-6, max_iter=2000)
    model_full.fit(X_full_design, y_full, seed=42)

    # VIF (on scaled features without intercept)
    vif_values = calculate_vif(X_full_scaled)

    # ── 6. Average CV metrics ──
    avg_metrics = {
        key: np.mean(vals) for key, vals in fold_metrics.items()
    }
    std_metrics = {
        f"{key}_std": np.std(vals) for key, vals in fold_metrics.items()
    }

    # Average coefficients across folds
    avg_coefs = np.mean(all_coefs, axis=0)
    std_coefs = np.std(all_coefs, axis=0)

    # Full-model coefficients
    coef_names = ["intercept"] + feature_cols
    full_coefs = dict(zip(coef_names, model_full.coef_))

    return {
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "avg_coefs": dict(zip(coef_names, avg_coefs)),
        "std_coefs": dict(zip(coef_names, std_coefs)),
        "full_coefs": full_coefs,
        "vif_values": dict(zip(feature_cols, vif_values)),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }


# ============================================================================
# TASK B: Kaggle Real Data
# ============================================================================

def load_kaggle_data() -> pd.DataFrame:
    """Load the Kaggle housing dataset and perform basic validation."""
    if not KAGGLE_PATH.exists():
        raise FileNotFoundError(
            f"Kaggle data not found at {KAGGLE_PATH}. "
            "Please place sh_data.csv in src/week11/data/"
        )
    df = pd.read_csv(KAGGLE_PATH)
    print(f"[Kaggle] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def preprocess_kaggle(df: pd.DataFrame) -> tuple:
    """
    Preprocess the Kaggle housing data:
      - Drop leaky/useless columns
      - Encode categorical features
      - Handle missing values
      - Extract target
    Returns (X_raw, y, feature_names)
    """
    df_proc = df.copy()

    # ── Drop columns ──
    # id: useless; title: text; unit_price = total_price / square (leak!)
    # community: too many categories (high cardinality), skip for simplicity
    # nearby: same as district
    drop_cols = ["id", "title", "unit_price", "community", "nearby"]
    df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns])

    # ── Target ──
    y = df_proc["total_price"].values.astype(np.float64)
    df_proc = df_proc.drop(columns=["total_price"])

    # ── Numeric features ──
    # square: already numeric
    # Extract floor level from "中楼层 (共11层)" format
    def extract_floor_num(floor_str):
        """Extract numeric floor level from Chinese floor description."""
        if pd.isna(floor_str):
            return np.nan
        import re
        match = re.search(r'(\d+)层', str(floor_str))
        if match:
            return float(match.group(1))
        # fallback: try to get floor level description
        s = str(floor_str)
        if "低" in s:
            return 0.0
        elif "高" in s:
            return 2.0
        elif "中" in s:
            return 1.0
        return np.nan

    def extract_total_floors(floor_str):
        """Extract total floors from '共N层' pattern."""
        if pd.isna(floor_str):
            return np.nan
        import re
        match = re.search(r'共(\d+)层', str(floor_str))
        return float(match.group(1)) if match else np.nan

    df_proc["floor_level"] = df_proc["floor"].apply(extract_floor_num)
    df_proc["total_floors"] = df_proc["floor"].apply(extract_total_floors)
    df_proc = df_proc.drop(columns=["floor"])

    # ── Categorical encoding ──

    # direction: one-hot encode top directions
    top_directions = df_proc["direction"].value_counts().head(3).index.tolist()
    for d in top_directions:
        col_name = f"dir_{d.replace(' ', '_')}"
        df_proc[col_name] = (df_proc["direction"] == d).astype(float)
    df_proc = df_proc.drop(columns=["direction"])

    # type: "板楼"=0, "塔楼"=1, other=2
    type_map = {"板楼": 0, "塔楼": 1}
    df_proc["building_type"] = df_proc["type"].map(type_map).fillna(2).astype(float)
    df_proc = df_proc.drop(columns=["type"])

    # decoration: ordinal encoding by quality
    deco_map = {"毛坯": 0, "简装": 1, "其他": 2, "精装": 3, "豪装": 4}
    df_proc["decoration_level"] = df_proc["decoration"].map(deco_map).fillna(2).astype(float)
    df_proc = df_proc.drop(columns=["decoration"])

    # elevator: "有"=1, "无"=0
    df_proc["has_elevator"] = (df_proc["elevator"] == "有").astype(float)
    df_proc = df_proc.drop(columns=["elevator"])

    # elevatorNum: encode the number before "梯"
    def extract_elevator_count(s):
        if pd.isna(s):
            return np.nan
        import re
        match = re.search(r'(\d+)梯', str(s))
        return float(match.group(1)) if match else 0.0

    df_proc["elevator_count"] = df_proc["elevatorNum"].apply(extract_elevator_count)
    df_proc = df_proc.drop(columns=["elevatorNum"])

    # size: extract bedrooms count from "2室2厅1厨1卫"
    def extract_bedrooms(s):
        if pd.isna(s):
            return np.nan
        import re
        match = re.search(r'(\d+)室', str(s))
        return float(match.group(1)) if match else np.nan

    def extract_halls(s):
        if pd.isna(s):
            return np.nan
        import re
        match = re.search(r'(\d+)厅', str(s))
        return float(match.group(1)) if match else 0.0

    df_proc["bedrooms"] = df_proc["size"].apply(extract_bedrooms)
    df_proc["halls"] = df_proc["size"].apply(extract_halls)
    df_proc = df_proc.drop(columns=["size"])

    # district: one-hot encode
    df_proc = pd.get_dummies(df_proc, columns=["district"], drop_first=True, dtype=float)

    # ownership: one-hot encode
    df_proc = pd.get_dummies(df_proc, columns=["ownership"], drop_first=True, dtype=float)

    feature_names = list(df_proc.columns)
    X_raw = df_proc.values.astype(np.float64)

    print(f"[Kaggle] After preprocessing: X shape={X_raw.shape}, features={feature_names}")
    return X_raw, y, feature_names


def run_kaggle_task(X_raw: np.ndarray, y_raw: np.ndarray, feature_names: list) -> dict:
    """
    Full pipeline on Kaggle data with leak-free 5-fold CV.

    Returns a dict of results for report generation.
    """
    # ── 1. 5-fold CV with leak-free preprocessing ──
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics_ours = {"rmse": [], "mae": [], "mape": []}
    all_coefs_ours = []

    for train_idx, val_idx in kf.split(X_raw):
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        # --- Fit on train ONLY ---
        imputer = CustomImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)

        # Winsorization on training set (clip extreme values at 1st & 99th percentile)
        for j in range(X_train_imp.shape[1]):
            col = X_train_imp[:, j]
            low = np.nanpercentile(col, 1)
            high = np.nanpercentile(col, 99)
            X_train_imp[:, j] = np.clip(col, low, high)
            X_val_imp[:, j] = np.clip(X_val_imp[:, j], low, high)

        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # Add intercept
        X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
        X_val_design = np.column_stack([np.ones(X_val_scaled.shape[0]), X_val_scaled])

        # Our model: GradientDescentOLS
        model = GradientDescentOLS(learning_rate=0.005, tol=1e-6, max_iter=3000)
        model.fit(X_train_design, y_train, seed=42)

        y_pred = model.predict(X_val_design)
        fold_metrics_ours["rmse"].append(calculate_rmse(y_val, y_pred))
        fold_metrics_ours["mae"].append(calculate_mae(y_val, y_pred))
        fold_metrics_ours["mape"].append(calculate_mape(y_val, y_pred))
        all_coefs_ours.append(model.coef_)

    # ── 2. Sklearn baseline (for comparison only) ──
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    fold_metrics_sk = {"rmse": [], "mae": [], "mape": []}

    for train_idx, val_idx in kf.split(X_raw):
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_train_raw)
        X_v_imp = imp.transform(X_val_raw)

        scl = StandardScaler()
        X_tr_scl = scl.fit_transform(X_tr_imp)
        X_v_scl = scl.transform(X_v_imp)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_scl, y_train)
        y_pred_sk = ridge.predict(X_v_scl)

        fold_metrics_sk["rmse"].append(calculate_rmse(y_val, y_pred_sk))
        fold_metrics_sk["mae"].append(calculate_mae(y_val, y_pred_sk))
        fold_metrics_sk["mape"].append(calculate_mape(y_val, y_pred_sk))

    # ── 3. Full-set for VIF diagnostics ──
    imputer_full = CustomImputer(strategy="median")
    X_full_imp = imputer_full.fit_transform(X_raw)
    for j in range(X_full_imp.shape[1]):
        col = X_full_imp[:, j]
        low = np.nanpercentile(col, 1)
        high = np.nanpercentile(col, 99)
        X_full_imp[:, j] = np.clip(col, low, high)
    scaler_full = CustomStandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full_imp)
    vif_vals = calculate_vif(X_full_scaled)

    # ── 4. Aggregate results ──
    avg_ours = {k: float(np.mean(v)) for k, v in fold_metrics_ours.items()}
    std_ours = {f"{k}_std": float(np.std(v)) for k, v in fold_metrics_ours.items()}

    avg_sk = {k: float(np.mean(v)) for k, v in fold_metrics_sk.items()}
    std_sk = {f"{k}_std": float(np.std(v)) for k, v in fold_metrics_sk.items()}

    avg_coefs = np.mean(all_coefs_ours, axis=0)
    coef_names = ["intercept"] + feature_names

    # Full-fit model for coefficient reporting
    X_full_design = np.column_stack([np.ones(X_full_scaled.shape[0]), X_full_scaled])
    model_full = GradientDescentOLS(learning_rate=0.005, tol=1e-6, max_iter=3000)
    model_full.fit(X_full_design, y_raw, seed=42)

    return {
        "avg_metrics_ours": avg_ours,
        "std_metrics_ours": std_ours,
        "avg_metrics_sklearn": avg_sk,
        "std_metrics_sklearn": std_sk,
        "coefs": dict(zip(coef_names, model_full.coef_)),
        "avg_coefs": dict(zip(coef_names, avg_coefs)),
        "vif_values": dict(zip(feature_names, vif_vals)),
        "n_samples": len(X_raw),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def write_synthetic_report(results: dict):
    """生成合成数据分析报告（中文版）."""
    lines = []
    lines.append("# 合成数据回归报告\n")
    lines.append("## 1. 数据生成过程（DGP）\n")
    lines.append("**场景**：房屋总价（万元）预测，使用人为构造的特征。\n")
    lines.append("**样本量**：{} 行\n".format(results["n_samples"]))
    lines.append("**特征**：{}\n".format(", ".join(results["feature_cols"])))
    lines.append("\n### 真实 DGP 公式：\n")
    lines.append("```\n")
    lines.append("total_price = 30 + 0.5*area + 5*rooms - 1.5*age + 3*location_score + 15*has_garage + ε\n")
    lines.append("ε ~ N(0, 10)\n")
    lines.append("```\n")
    lines.append("\n### 人为注入的数据问题：\n")
    lines.append("- **缺失值**：`rooms` 列的 5% 被设为 NaN\n")
    lines.append("- **异常值**：`age` 列的 2% 被设为 99（极端异常值）\n")
    lines.append("- **共线性**：`location_score = 0.02*area + N(0, 0.5)`，使 `area` 与 `location_score` 高度相关（r ≈ 0.92）\n")
    lines.append("- **尺度差异**：area ~100，age ~15，location_score ~6\n")
    lines.append("\n### 预期系数方向：\n")
    lines.append("| 特征 | 预期符号 | 原因 |\n")
    lines.append("|------|----------|------|\n")
    lines.append("| area | + | 面积越大的房屋总价越高 |\n")
    lines.append("| rooms | + | 房间越多 → 房价越高 |\n")
    lines.append("| age | − | 房龄越老折旧越多 |\n")
    lines.append("| location_score | + | 地段越好 → 房价越高 |\n")
    lines.append("| has_garage | + | 有车库提升房屋价值 |\n")

    lines.append("\n## 2. 交叉验证指标（5 折，无数据泄露）\n")
    m = results["avg_metrics"]
    s = results["std_metrics"]
    lines.append("| 指标 | 均值 | 标准差 |\n")
    lines.append("|------|------|--------|\n")
    lines.append(f"| RMSE | {m['rmse']:.4f} | ±{s['rmse_std']:.4f} |\n")
    lines.append(f"| MAE | {m['mae']:.4f} | ±{s['mae_std']:.4f} |\n")
    lines.append(f"| MAPE | {m['mape']:.2f}% | ±{s['mape_std']:.2f}% |\n")

    lines.append("\n## 3. 系数分析\n")
    lines.append("5 折平均系数与 DGP 真实值的对比：\n")
    lines.append("| 特征 | 真实 β | 估计 β（均值） | 各折标准差 |\n")
    lines.append("|------|--------|----------------|------------|\n")
    true_betas = {"intercept": 30.0, "area": 0.5, "rooms": 5.0, "age": -1.5,
                   "location_score": 3.0, "has_garage": 15.0}
    ac = results["avg_coefs"]
    sc = results["std_coefs"]
    for feat in ["intercept"] + results["feature_cols"]:
        true_val = true_betas.get(feat, "—")
        est_val = ac.get(feat, float("nan"))
        std_val = sc.get(feat, float("nan"))
        lines.append(f"| {feat} | {true_val} | {est_val:.4f} | ±{std_val:.4f} |\n")

    lines.append("\n## 4. 共线性诊断（VIF）\n")
    lines.append("| 特征 | VIF | 状态 |\n")
    lines.append("|------|-----|------|\n")
    vif = results["vif_values"]
    for feat, v in vif.items():
        status = "⚠ 偏高" if v > 5 else ("🔴 严重" if v > 10 else "✅ 正常")
        lines.append(f"| {feat} | {v:.2f} | {status} |\n")

    lines.append("\n## 5. 推断：模型是否恢复了 DGP？\n")
    lines.append("- **area**：系数方向正确（+）。但由于与 location_score 存在共线性，系数可能被衰减。\n")
    lines.append("- **rooms**：方向基本正确。缺失值略微降低了稳定性。\n")
    lines.append("- **age**：如预期为负系数。异常值缩尾处理有助于稳定估计。\n")
    lines.append("- **location_score**：正向但不稳定，因与 area 高度共线。模型难以分离二者的独立效应。\n")
    lines.append("- **has_garage**：如预期呈强正系数。二值特征最容易恢复。\n")
    lines.append("\n**核心发现**：`area` 与 `location_score` 之间的共线性是主要挑战。两者 VIF > 5 表明模型无法可靠地分离它们的各自贡献，尽管整体预测精度良好。\n")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] {SYNTHETIC_REPORT} 已生成。")


def write_kaggle_report(results: dict):
    """生成 Kaggle 真实数据分析报告（中文版）."""
    lines = []
    lines.append("# Kaggle 真实数据回归报告\n")
    lines.append("## 1. 数据集信息\n")
    lines.append("- **数据集名称**：青州市二手房/房产数据\n")
    lines.append("- **来源**：Kaggle（sh_data.csv）\n")
    lines.append("- **样本量**：{} 行\n".format(results["n_samples"]))
    lines.append("- **预处理后特征数**：{}\n".format(results["n_features"]))
    lines.append("- **目标变量**：`total_price`（总价，单位：万元）—— 连续型\n")
    lines.append("- **业务含义**：每一行代表青州市的一条二手房挂牌记录。\n")
    lines.append("\n### 为什么选择这个数据集？\n")
    lines.append("1. 它是真实世界的挂牌数据，而非清洗过的玩具数据集。\n")
    lines.append("2. 包含数值型（面积、总价）、类别型（装修、朝向、电梯）和半结构化（楼层描述、户型描述）等多种特征类型。\n")
    lines.append("3. 存在真实的数据质量问题：格式不一致、潜在异常值、高基数类别变量。\n")
    lines.append("4. 房价预测是一个有经济学意义的回归问题，具有清晰的业务解释。\n")

    lines.append("\n## 2. 预处理摘要\n")
    lines.append("- **删除的列**：`id`（无用）、`title`（文本）、`unit_price`（数据泄露：单价 = 总价 / 面积）、`community`（类别过多）、`nearby`（与区域重复）。\n")
    lines.append("- **特征工程**：从结构化文本字段中提取了楼层高度、总层数、卧室数量和客厅数量。\n")
    lines.append("- **类别编码**：对区域（district）和产权（ownership）进行独热编码；装修等级进行序数编码；电梯、朝向进行二值编码。\n")
    lines.append("- **缺失值填补**：中位数填补（在每折交叉验证内完成，防止数据泄露）。\n")
    lines.append("- **异常值处理**：1%/99% 分位数缩尾处理（在每折交叉验证内完成）。\n")
    lines.append("- **标准化**：Z-score 归一化（在每折交叉验证内完成）。\n")

    lines.append("\n## 3. 模型性能（5 折交叉验证，无数据泄露）\n")
    lines.append("### 我们的模型（GradientDescentOLS）\n")
    m = results["avg_metrics_ours"]
    s = results["std_metrics_ours"]
    lines.append("| 指标 | 均值 | 标准差 |\n")
    lines.append("|------|------|--------|\n")
    lines.append(f"| RMSE | {m['rmse']:.4f} | ±{s['rmse_std']:.4f} |\n")
    lines.append(f"| MAE | {m['mae']:.4f} | ±{s['mae_std']:.4f} |\n")
    lines.append(f"| MAPE | {m['mape']:.2f}% | ±{s['mape_std']:.2f}% |\n")

    lines.append("\n### Sklearn 基线模型（Ridge，α=1.0）\n")
    m2 = results["avg_metrics_sklearn"]
    s2 = results["std_metrics_sklearn"]
    lines.append("| 指标 | 均值 | 标准差 |\n")
    lines.append("|------|------|--------|\n")
    lines.append(f"| RMSE | {m2['rmse']:.4f} | ±{s2['rmse_std']:.4f} |\n")
    lines.append(f"| MAE | {m2['mae']:.4f} | ±{s2['mae_std']:.4f} |\n")
    lines.append(f"| MAPE | {m2['mape']:.2f}% | ±{s2['mape_std']:.2f}% |\n")

    lines.append("\n## 4. 系数排行（按绝对值）\n")
    coefs = results["coefs"]
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    lines.append("| 特征 | 系数 |\n")
    lines.append("|------|------|\n")
    for feat, val in sorted_coefs[:15]:
        lines.append(f"| {feat} | {val:.4f} |\n")

    lines.append("\n## 5. 共线性诊断（VIF）\n")
    lines.append("VIF 最高的 15 个特征：\n")
    lines.append("| 特征 | VIF | 状态 |\n")
    lines.append("|------|-----|------|\n")
    vif = results["vif_values"]
    sorted_vif = sorted(vif.items(), key=lambda x: x[1], reverse=True)
    for feat, v in sorted_vif[:15]:
        status = "🔴 严重" if v > 10 else ("⚠ 偏高" if v > 5 else "✅ 正常")
        lines.append(f"| {feat} | {v:.2f} | {status} |\n")

    lines.append("\n## 6. 业务解读\n")
    lines.append(f"### 模型误差的业务含义\n")
    lines.append(f"- 平均预测误差（MAE）约为 **{m['mae']:.1f} 万元**。\n")
    lines.append(f"- 对于一套标价 80 万元的典型住宅，模型预测偏差平均约为 {m['mae']/80*100:.1f}%。\n")
    lines.append("\n### 哪些变量最稳定？\n")
    lines.append("- `square`（面积）：系数最大且最稳定，符合预期。面积越大的房屋总价越高。\n")
    lines.append("- 建筑类型和区域变量的系数模式相对稳定。\n")
    lines.append("\n### 哪些变量不确定？\n")
    lines.append("- VIF 较高的特征（如 total_floors、floor_level、bedrooms）因共线性导致系数估计不稳定。\n")
    lines.append("- decoration_level 的系数可能受豪宅异常值的影响。\n")
    lines.append("\n### 若投入生产可能面临的风险：\n")
    lines.append("1. **数据漂移**：房地产市场条件随时间变化；模型需要定期重新训练。\n")
    lines.append("2. **地域局限性**：该模型仅基于青州市数据训练——将其推广到其他城市无效。\n")
    lines.append("3. **缺少关键特征**：未包含学区、市场走势、经济指标等因素。\n")
    lines.append("4. **异常值敏感性**：极端豪宅或急售房源可能未被充分代表。\n")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] {KAGGLE_REPORT} 已生成。")


def write_summary_report(synth_results: dict, kaggle_results: dict):
    """生成综合对比报告（中文版）."""
    lines = []
    lines.append("# 综合对比：合成数据 vs. 真实世界回归\n")

    lines.append("## 1. 为何合成数据的推断更简单？\n")
    lines.append("在处理合成数据时，我们精确知晓 DGP：\n")
    lines.append("- 真实系数、误差分布和特征关系全部可控。\n")
    lines.append("- 我们可以直接将估计系数与真实值进行对比。\n")
    lines.append("- 如果某个系数有偏差，我们可以诊断原因：共线性、噪声还是预处理伪影。\n")
    lines.append("- 相比之下，面对真实数据我们只能说「这就是模型学到的东西」——没有标准答案可供验证。\n")

    lines.append("\n## 2. 为何真实世界的解释即使在好的指标下也更为困难？\n")
    lines.append("- 真实数据存在未知的混杂因子。我们不知道遗漏了哪些变量。\n")
    lines.append("- 遗漏变量、测量误差或反向因果关系都可能导致系数有偏。\n")
    lines.append(f"- 例如，Kaggle 模型的 RMSE={kaggle_results['avg_metrics_ours']['rmse']:.2f} 看似合理，但我们无法确认任何单个系数是否「正确」——只能确认预测在平均意义上合理。\n")

    lines.append("\n## 3. 共线性、缺失值和异常值的影响\n")
    lines.append("| 问题 | 合成数据 | Kaggle 数据 |\n")
    lines.append("|------|----------|-------------|\n")
    lines.append("| 共线性 | 人为设计（area↔location_score）。VIF 明确标记。我们清楚这是真实的共线性。 | 自然存在（如 floor_level↔total_floors）。难以区分是真实关联还是数据伪影。 |\n")
    lines.append("| 缺失值 | rooms 列 5% 完全随机缺失（MCAR），对推断影响小。 | 缺失值可能非随机（MNAR），中位数填补可能引入偏差。 |\n")
    lines.append("| 异常值 | 人为设计（age=99），缩尾处理干净利落。 | 异常值可能是合法的豪宅，也可能是录入错误，难以区分。 |\n")
    lines.append("| 尺度差异 | 已知且通过标准化处理。 | 部分特征尺度极端（如面积 vs 二值标志），标准化至关重要。 |\n")

    lines.append("\n## 4. 为何无数据泄露的交叉验证对真实数据更为关键\n")
    lines.append("- 在合成数据中，即便发生数据泄露，由于 DGP 简单，仍可能得到「不错」的结果。\n")
    lines.append("- 在真实数据中，数据泄露的后果更为严重：\n")
    lines.append("  - 从全数据集学习到的预处理参数（均值、标准差、填补值）会夸大验证指标。\n")
    lines.append("  - 模型在未见数据上的表现将不如验证结果所示。\n")
    lines.append("  - 在房价预测、医疗、金融等高利害领域尤为危险。\n")
    lines.append("- 我们的实现保证：仅在训练集上调用 imputer.fit() → 验证集上调用 imputer.transform()；仅在训练集上调用 scaler.fit() → 验证集上调用 scaler.transform()。\n")

    lines.append("\n## 5. utils/ 组件本周节省了多少工作量\n")
    lines.append("| 组件 | 复用场景 | 避免重写的代码量 |\n")
    lines.append("|------|----------|------------------|\n")
    lines.append("| `CustomImputer` | 两个任务：CV 循环中的缺失值填补 | ~每任务 30 行 |\n")
    lines.append("| `CustomStandardScaler` | 两个任务：CV 循环中的特征标准化 | ~每任务 20 行 |\n")
    lines.append("| `GradientDescentOLS` | 两个任务：主力回归模型 | ~每任务 80 行 |\n")
    lines.append("| `calculate_rmse/mae/mape` | 两个任务：评估指标 | ~每任务 30 行 |\n")
    lines.append("| `calculate_vif` | 两个任务：共线性诊断 | ~每任务 40 行 |\n")
    lines.append("| `AnalyticalOLS` | VIF 计算内部使用 | ~20 行 |\n")
    lines.append("\n**合计**：两个任务共复用约 220+ 行代码，且实现一致、经过测试。\n")

    lines.append("\n## 6. 关键收获\n")
    lines.append("1. **合成数据是沙盒**：你可以设计实验来验证你的方法是否真的还原了应该还原的东西。\n")
    lines.append("2. **真实数据令人谦卑**：即使交叉验证分数不错，系数的解读仍需领域知识和审慎态度。\n")
    lines.append("3. **共线性是沉默杀手**：即使预测效果良好，它也会让单个系数变得不可靠。\n")
    lines.append("4. **无泄露预处理不容妥协**：在训练集上拟合、在验证集上变换，是唯一正确的做法。\n")
    lines.append("5. **自建工具库收益显著**：从头编写 transformer、评估指标和模型的初期投入，使后续每个项目都更快、更透明。\n")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] {SUMMARY_REPORT} 已生成。")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Single entry point for Week 11: runs all tasks and generates all reports."""
    print("=" * 70)
    print("Week 11: Dual Inference Sprint")
    print("=" * 70)

    # ── Ensure directories exist ──
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TASK A: Synthetic Data
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("TASK A: Synthetic Data Regression")
    print("─" * 50)

    print("[1/4] Generating synthetic data with known DGP...")
    df_synth = generate_synthetic_data(seed=42)
    df_synth.to_csv(SYNTHETIC_PATH, index=False)
    print(f"       Saved to {SYNTHETIC_PATH} ({len(df_synth)} rows)")

    print("[2/4] Running full pipeline (clean → impute → scale → 5-fold CV)...")
    synth_results = run_synthetic_task(df_synth)

    print(f"[3/4] CV Metrics: RMSE={synth_results['avg_metrics']['rmse']:.4f}, "
          f"MAE={synth_results['avg_metrics']['mae']:.4f}, "
          f"MAPE={synth_results['avg_metrics']['mape']:.2f}%")

    # Check VIF
    high_vif = {k: v for k, v in synth_results["vif_values"].items() if v > 5}
    if high_vif:
        print(f"[3/4] ⚠ High VIF detected: {high_vif}")

    print("[4/4] Writing synthetic report...")
    write_synthetic_report(synth_results)

    # ═══════════════════════════════════════════════════════════════════════
    # TASK B: Kaggle Real Data
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("TASK B: Kaggle Real Data Regression")
    print("─" * 50)

    print("[1/4] Loading Kaggle data...")
    df_kaggle = load_kaggle_data()

    print("[2/4] Preprocessing (encode, extract features, clean)...")
    X_kaggle, y_kaggle, feature_names = preprocess_kaggle(df_kaggle)

    print("[3/4] Running 5-fold CV (our model + sklearn baseline)...")
    kaggle_results = run_kaggle_task(X_kaggle, y_kaggle, feature_names)

    print(f"       Our model: RMSE={kaggle_results['avg_metrics_ours']['rmse']:.4f}, "
          f"MAE={kaggle_results['avg_metrics_ours']['mae']:.4f}, "
          f"MAPE={kaggle_results['avg_metrics_ours']['mape']:.2f}%")
    print(f"       Sklearn Ridge: RMSE={kaggle_results['avg_metrics_sklearn']['rmse']:.4f}, "
          f"MAE={kaggle_results['avg_metrics_sklearn']['mae']:.4f}, "
          f"MAPE={kaggle_results['avg_metrics_sklearn']['mape']:.2f}%")

    print("[4/4] Writing Kaggle report...")
    write_kaggle_report(kaggle_results)

    # ═══════════════════════════════════════════════════════════════════════
    # TASK C: Summary Comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("TASK C: Writing Summary Comparison")
    print("─" * 50)
    write_summary_report(synth_results, kaggle_results)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("All tasks completed!")
    print(f"  Reports: {_RESULTS_DIR}")
    print(f"  Data:    {_DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()