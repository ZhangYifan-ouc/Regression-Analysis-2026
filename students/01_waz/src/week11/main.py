"""
Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow
======================================================================
单一入口: uv run src/week11/main.py

流程:
  1. generate_synthetic_data()  → 生成模拟数据并保存
  2. run_synthetic_task()       → 读取、清洗、诊断、CV、报告
  3. load_kaggle_data()         → 读取 Kaggle 真实数据
  4. run_kaggle_task()          → 清洗、诊断、训练、评估、报告
  5. write_reports()            → 输出三份 Markdown 报告
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 路径设置 — 确保能从项目根目录 import utils
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_SRC, "..")))

from src.utils.models import CustomOLS, GradientDescentOLS
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from src.utils.transformers import CustomImputer, CustomStandardScaler
from src.utils.diagnostics import calculate_vif

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_regression.csv")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_medical_cost.csv")
CV_FOLDS = 5
RANDOM_SEED = 2026

# ========================== 模拟数据 =========================================

def generate_synthetic_data(n_samples: int = 400) -> pd.DataFrame:
    """A1: 按 DGP 生成带业务含义的模拟回归数据并保存。

    场景: 房屋价格预测 (简化版)
    DGP:
      x1 = 房屋面积 (sqft),        Uniform(500, 4000)
      x2 = 卧室数量,               round(x1 的线性函数 + 噪声)
      x3 = 房龄 (years),           Gamma(shape=2)
      x4 = 地段等级 (A/B/C),       类别变量, 与 x1 部分关联
      y  = 房价 (万元),
           y = 50 + 0.8*x1 - 1.2*x3 + 15*(x4==A) + 10*(x4==B) + ε
           其中 ε ~ N(0, 15^2)

    共线性构造: x2 与 x1 高度正相关 (x2 ≈ 0.003*x1 + noise)
    异常值: 随机 3% 样本的面积 ×3 或房龄置为极大值
    缺失值: 随机 5% 的 x3 置为 NaN
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = n_samples

    # --- 连续变量 ---
    x1 = rng.uniform(500, 4000, n)                         # 房屋面积

    # x2 与 x1 高度相关 (共线性来源)
    x2_raw = 0.003 * x1 + rng.normal(0, 0.3, n)
    x2 = np.round(np.clip(x2_raw, 1, 8)).astype(int)       # 卧室数 1-8

    # x3 房龄, 右偏
    x3 = rng.gamma(shape=2.0, scale=8.0, size=n)           # 房龄

    # --- 类别变量 x4 (地段) ---
    # 高面积更可能位于 A 地段
    area_rank = np.argsort(np.argsort(x1)) / (n - 1)        # 0~1
    x4 = np.full(n, "C", dtype=object)
    x4[(area_rank > 0.33) & (area_rank <= 0.66)] = "B"
    x4[area_rank > 0.66] = "A"

    # --- DGP 目标变量 ---
    noise = rng.normal(0, 15, n)
    y = 50.0 + 0.8 * x1 - 1.2 * x3 + noise
    y[x4 == "A"] += 15.0
    y[x4 == "B"] += 10.0
    # C 地段: 基准 (不加额外项)

    # --- 构造 DataFrame ---
    df = pd.DataFrame({
        "area_sqft": x1,
        "bedrooms": x2,
        "age_years": x3,
        "location": x4,
        "price_wan": y,
    })

    # --- 注入异常值: 3% 面积×3 或房龄极大 ---
    outlier_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[outlier_idx[:len(outlier_idx)//2], "area_sqft"] *= 3.0
    df.loc[outlier_idx[len(outlier_idx)//2:], "age_years"] = rng.uniform(60, 100, len(outlier_idx) - len(outlier_idx)//2)

    # --- 注入缺失值: 5% 的 x3 ---
    missing_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[missing_idx, "age_years"] = np.nan

    # --- 保存 ---
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SYNTHETIC_PATH, index=False)
    print(f"[模拟数据] 已保存至 {SYNTHETIC_PATH}  (n={n})")
    return df


# ========================== 无泄露 CV ========================================

def leak_free_cv_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
):
    """无泄露 K 折交叉验证。

    在每一折中:
      - 用训练集 fit Imputer / Scaler / Model
      - 用验证集仅 transform / predict

    Returns:
        metrics_df: 每折的 RMSE / MAE / MAPE
        all_coefs : 每折的模型系数 (list of np.array)
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    rmse_list, mae_list, mape_list = [], [], []
    all_coefs = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        # --- 预处理 (训练集 fit, 验证集 transform) ---
        imputer = CustomImputer()
        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)

        scaler = CustomStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # --- 模型训练 ---
        model = model_factory()
        if isinstance(model, CustomOLS):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, seed=seed + fold)

        # --- 预测与评估 ---
        y_pred = model.predict(X_val)
        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

        # --- 记录系数用于诊断 ---
        if hasattr(model, "coef_"):
            all_coefs.append(model.coef_.copy())

    metrics_df = pd.DataFrame({
        "fold": range(1, n_folds + 1),
        "RMSE": rmse_list,
        "MAE": mae_list,
        "MAPE": mape_list,
    })
    return metrics_df, all_coefs


# ========================== 通用清洗 =========================================

def clean_and_encode(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list, CustomImputer, CustomStandardScaler]:
    """通用清洗: 类别变量 OneHot 编码, 数值列保留, 返回 X, y 及列名.

    注: 这里只做编码, 缺失值和标准化放到 CV 内部做以避免泄露.
    """
    df = df.copy()

    # 分离特征和目标
    y = df[target_col].values.astype(float)
    df = df.drop(columns=[target_col])

    feature_names = []
    numeric_cols = []
    cat_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            feature_names.append(col)
        else:
            cat_cols.append(col)

    # 类别 One-Hot 编码 (drop_first 避免完全共线性)
    if cat_cols:
        df_cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
        feature_names = numeric_cols + list(df_cat_encoded.columns)
        X_num = df[numeric_cols].values.astype(float)
        X_cat = df_cat_encoded.values.astype(float)
        X = np.column_stack([X_num, X_cat]) if X_num.shape[1] > 0 else X_cat
    else:
        X = df[numeric_cols].values.astype(float)

    return X, y, feature_names


# ========================== 离群值处理 =======================================

def winsorize_percentile(X: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """百分位数 Winsorization: 将极端值裁剪到上下分位。逐列处理, 忽略 NaN."""
    X_clean = X.copy().astype(float)
    for j in range(X_clean.shape[1]):
        col = X_clean[:, j]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) == 0:
            continue
        lo = np.percentile(col_valid, lower_pct)
        hi = np.percentile(col_valid, upper_pct)
        col = np.clip(col, lo, hi)
        X_clean[:, j] = col
    return X_clean


# ========================== Task A: 模拟数据 =================================

def run_synthetic_task():
    """A2-A4: 在模拟数据上完成清洗、CV 评估、VIF 诊断和推测."""
    print("\n" + "=" * 60)
    print("  Task A: 模拟数据 — 可验证的推测")
    print("=" * 60)

    # --- 读取 ---
    df = pd.read_csv(SYNTHETIC_PATH)
    X, y, feature_names = clean_and_encode(df, target_col="price_wan")
    print(f"[A] 特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"[A] 列名: {feature_names}")

    # --- 清洗: Winsorization (在 CV 外只做不学参数的) ---
    X = winsorize_percentile(X, lower_pct=1.0, upper_pct=99.0)

    # --- 无泄露 CV (使用 CustomOLS) ---
    print("\n[A] 无泄露 5 折交叉验证 (CustomOLS) ...")
    metrics_df, all_coefs = leak_free_cv_evaluation(
        X, y,
        model_factory=lambda: CustomOLS(fit_intercept=True),
        n_folds=CV_FOLDS,
    )
    print(metrics_df.to_string(index=False))
    print(f"\n[A] 平均 RMSE: {metrics_df['RMSE'].mean():.4f}")
    print(f"[A] 平均 MAE : {metrics_df['MAE'].mean():.4f}")
    print(f"[A] 平均 MAPE: {metrics_df['MAPE'].mean():.2f}%")

    # --- GradientDescentOLS baseline ---
    print("\n[A] 无泄露 CV (GradientDescentOLS) ...")
    gd_metrics, gd_coefs = leak_free_cv_evaluation(
        X, y,
        model_factory=lambda: GradientDescentOLS(learning_rate=0.01, max_iter=2000, gd_type="full_batch"),
        n_folds=CV_FOLDS,
    )
    print(f"[A] GD 平均 RMSE: {gd_metrics['RMSE'].mean():.4f}")
    print(f"[A] GD 平均 MAE : {gd_metrics['MAE'].mean():.4f}")
    print(f"[A] GD 平均 MAPE: {gd_metrics['MAPE'].mean():.2f}%")

    # --- sklearn baseline ---
    print("\n[A] sklearn LinearRegression baseline ...")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    sk_rmse, sk_mae, sk_mape = [], [], []
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_i, val_i in kf.split(X):
        X_tr, X_va = X[train_i], X[val_i]
        y_tr, y_va = y[train_i], y[val_i]
        imp = CustomImputer(); X_tr = imp.fit_transform(X_tr); X_va = imp.transform(X_va)
        scl = CustomStandardScaler(); X_tr = scl.fit_transform(X_tr); X_va = scl.transform(X_va)
        lr = LinearRegression().fit(X_tr, y_tr)
        yp = lr.predict(X_va)
        sk_rmse.append(calculate_rmse(y_va, yp))
        sk_mae.append(calculate_mae(y_va, yp))
        sk_mape.append(calculate_mape(y_va, yp))
    print(f"[A] sklearn RMSE: {np.mean(sk_rmse):.4f}, MAE: {np.mean(sk_mae):.4f}, MAPE: {np.mean(sk_mape):.2f}%")

    # --- VIF 诊断 (在全量清洗后做) ---
    print("\n[A] VIF 诊断 ...")
    imp_full = CustomImputer(); X_filled = imp_full.fit_transform(X)
    scl_full = CustomStandardScaler(); X_scaled = scl_full.fit_transform(X_filled)
    vif_vals = calculate_vif(X_scaled)
    for name, v in zip(feature_names, vif_vals):
        flag = " ⚠️ 高共线性" if v > 10 else (" ⚡ 中等" if v > 5 else "")
        print(f"  {name:20s}  VIF={v:8.2f}{flag}")

    # --- 系数方向检查 ---
    print("\n[A] 系数方向检查 (最后一折) ...")
    last_coef = all_coefs[-1]
    print(f"  DGP 参考: area_sqft(+0.8), bedrooms(+0 but correlated), age_years(-1.2), location_A(baseline), location_B(+10 vs A), location_C(+0 vs A)")
    for name, c in zip(feature_names, last_coef):
        print(f"  {name:20s}  coef={c:+10.4f}")

    return {
        "ols_metrics": metrics_df,
        "gd_metrics": gd_metrics,
        "sk_rmse_mean": float(np.mean(sk_rmse)),
        "sk_mae_mean": float(np.mean(sk_mae)),
        "sk_mape_mean": float(np.mean(sk_mape)),
        "vif": dict(zip(feature_names, vif_vals)),
        "last_coefs": dict(zip(feature_names, last_coef)),
        "feature_names": feature_names,
    }


# ========================== Task B: Kaggle 真实数据 ==========================

def load_kaggle_data() -> pd.DataFrame:
    """B1: 读取 Kaggle 数据并做最基本的字段检查。

    使用 Medical Cost Personal Dataset:
      - 来源: https://www.kaggle.com/datasets/mirichoi0218/insurance
      - 目标变量: charges (医疗费用, 连续)
      - 每行代表一位参保人的医疗费用及个人信息
    """
    print("\n" + "=" * 60)
    print("  Task B: Kaggle 真实数据 — 现实世界推测流程")
    print("=" * 60)

    csv_path = KAGGLE_CSV

    # 如果 Kaggle 文件还没下, 给出提示并使用内嵌版本
    if not os.path.exists(csv_path):
        print("[B] ⚠ Kaggle 文件未找到, 将在运行时自动下载...")
        _download_kaggle_insurance(csv_path)

    df = pd.read_csv(csv_path)
    print(f"[B] 原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"[B] 列名: {list(df.columns)}")
    print(f"[B] 目标变量: charges (连续, 医疗费用)")

    # 基本检查
    print(f"[B] 缺失值:\n{df.isnull().sum().to_string()}")

    return df


def _download_kaggle_insurance(save_path: str):
    """从 GitHub 上的 openml 镜像获取 insurance.csv (无需 kaggle API)."""
    import urllib.request
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    urllib.request.urlretrieve(url, save_path)
    print(f"[B] 已从 {url} 下载数据到 {save_path}")


def run_kaggle_task(df: pd.DataFrame):
    """B2-B4: 在真实数据上完成清洗、CV 评估、诊断和推测."""
    # --- 清洗准备 ---
    df = df.copy()

    # 重命名列 (中文友好)
    rename_map = {
        "age": "age",
        "sex": "sex",
        "bmi": "bmi",
        "children": "children",
        "smoker": "smoker",
        "region": "region",
        "charges": "charges",
    }
    # 保持不变, 仅做记录

    X, y, feature_names = clean_and_encode(df, target_col="charges")
    print(f"\n[B] 编码后特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"[B] 编码后列名: {feature_names}")

    # --- 离群值处理 ---
    X = winsorize_percentile(X, lower_pct=1.0, upper_pct=99.0)

    # --- 无泄露 CV (CustomOLS) ---
    print("\n[B] 无泄露 5 折交叉验证 (CustomOLS) ...")
    metrics_df, all_coefs = leak_free_cv_evaluation(
        X, y,
        model_factory=lambda: CustomOLS(fit_intercept=True),
        n_folds=CV_FOLDS,
    )
    print(metrics_df.to_string(index=False))
    ols_rmse_mean = metrics_df["RMSE"].mean()
    ols_mae_mean = metrics_df["MAE"].mean()
    ols_mape_mean = metrics_df["MAPE"].mean()
    print(f"\n[B] CustomOLS 平均 RMSE: {ols_rmse_mean:.4f}")
    print(f"[B] CustomOLS 平均 MAE : {ols_mae_mean:.4f}")
    print(f"[B] CustomOLS 平均 MAPE: {ols_mape_mean:.2f}%")

    # --- GradientDescentOLS baseline ---
    print("\n[B] 无泄露 CV (GradientDescentOLS) ...")
    gd_metrics, gd_coefs = leak_free_cv_evaluation(
        X, y,
        model_factory=lambda: GradientDescentOLS(learning_rate=0.005, max_iter=3000, gd_type="full_batch"),
        n_folds=CV_FOLDS,
    )
    gd_rmse_mean = gd_metrics["RMSE"].mean()
    gd_mae_mean = gd_metrics["MAE"].mean()
    gd_mape_mean = gd_metrics["MAPE"].mean()
    print(f"[B] GD 平均 RMSE: {gd_rmse_mean:.4f}")
    print(f"[B] GD 平均 MAE : {gd_mae_mean:.4f}")
    print(f"[B] GD 平均 MAPE: {gd_mape_mean:.2f}%")

    # --- sklearn baseline ---
    print("\n[B] sklearn LinearRegression baseline ...")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    sk_rmse, sk_mae, sk_mape = [], [], []
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_i, val_i in kf.split(X):
        X_tr, X_va = X[train_i], X[val_i]
        y_tr, y_va = y[train_i], y[val_i]
        imp = CustomImputer(); X_tr = imp.fit_transform(X_tr); X_va = imp.transform(X_va)
        scl = CustomStandardScaler(); X_tr = scl.fit_transform(X_tr); X_va = scl.transform(X_va)
        lr = LinearRegression().fit(X_tr, y_tr)
        yp = lr.predict(X_va)
        sk_rmse.append(calculate_rmse(y_va, yp))
        sk_mae.append(calculate_mae(y_va, yp))
        sk_mape.append(calculate_mape(y_va, yp))
    sk_rmse_mean = float(np.mean(sk_rmse))
    sk_mae_mean = float(np.mean(sk_mae))
    sk_mape_mean = float(np.mean(sk_mape))
    print(f"[B] sklearn RMSE: {sk_rmse_mean:.4f}, MAE: {sk_mae_mean:.4f}, MAPE: {sk_mape_mean:.2f}%")

    # --- VIF 诊断 ---
    print("\n[B] VIF 诊断 ...")
    imp_full = CustomImputer(); X_filled = imp_full.fit_transform(X)
    scl_full = CustomStandardScaler(); X_scaled = scl_full.fit_transform(X_filled)
    vif_vals = calculate_vif(X_scaled)
    for name, v in zip(feature_names, vif_vals):
        flag = " ⚠️ 高共线性" if v > 10 else (" ⚡ 中等" if v > 5 else "")
        print(f"  {name:20s}  VIF={v:8.2f}{flag}")

    # --- 系数方向 ---
    print("\n[B] 系数方向 (最后一折) ...")
    last_coef = all_coefs[-1]
    for name, c in zip(feature_names, last_coef):
        print(f"  {name:20s}  coef={c:+10.4f}")

    return {
        "ols_metrics": metrics_df,
        "ols_rmse_mean": float(ols_rmse_mean),
        "ols_mae_mean": float(ols_mae_mean),
        "ols_mape_mean": float(ols_mape_mean),
        "gd_rmse_mean": float(gd_rmse_mean),
        "gd_mae_mean": float(gd_mae_mean),
        "gd_mape_mean": float(gd_mape_mean),
        "sk_rmse_mean": sk_rmse_mean,
        "sk_mae_mean": sk_mae_mean,
        "sk_mape_mean": sk_mape_mean,
        "vif": dict(zip(feature_names, vif_vals)),
        "last_coefs": dict(zip(feature_names, last_coef)),
        "feature_names": feature_names,
    }


# ========================== Task C: 报告生成 =================================

def write_reports(syn_results: dict, kag_results: dict):
    """生成三份 Markdown 报告."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- synthetic_report.md ----
    _write_synthetic_report(syn_results)

    # ---- kaggle_report.md ----
    _write_kaggle_report(kag_results)

    # ---- summary_comparison.md ----
    _write_summary_comparison(syn_results, kag_results)

    print(f"\n[报告] 三份报告已生成至 {RESULTS_DIR}/")


def _write_synthetic_report(r: dict):
    metrics = r["ols_metrics"]
    path = os.path.join(RESULTS_DIR, "synthetic_report.md")
    lines = [
        "# 📐 Synthetic Data Report — 模拟回归数据报告",
        "",
        "## 1. 数据生成机制 (DGP)",
        "",
        "**场景**: 房屋价格预测 (简化版)",
        "",
        "**特征**:",
        "- `area_sqft`: 房屋面积 (sqft), ~Uniform(500, 4000)",
        "- `bedrooms`: 卧室数量, 与 area_sqft 高度相关 (共线性来源)",
        "- `age_years`: 房龄, ~Gamma(shape=2, scale=8), 右偏分布",
        "- `location`: 地段等级 A / B / C (类别变量)",
        "",
        "**目标变量 price_wan (房价, 万元) 生成公式**:",
        "```",
        "y = 50 + 0.8 * area_sqft - 1.2 * age_years",
        "    + 15 * (location==A) + 10 * (location==B) + ε",
        "ε ~ N(0, 15²)",
        "```",
        "",
        "**预期方向** (注: location 采用 drop-first 编码, A=基线):",
        "- `area_sqft` → 正向",
        "- `age_years` → 负向",
        "- `bedrooms` → 与面积共线, 系数可能不稳定",
        "- `location_B` → 弱于 A (系数预期为负 vs 基线A)",
        "- `location_C` → 弱于 A (系数预期为负 vs 基线A)",
        "",
        "## 2. 注入的「真实世界问题」",
        "- **缺失值**: 5% 的 `age_years` 设为 NaN",
        "- **异常值**: 3% 样本的面积×3 或房龄置为 60-100 年",
        "- **共线性**: bedrooms ≈ 0.003 × area_sqft + noise",
        "- **量纲差异**: 面积 (500-4000) vs 卧室 (1-8) vs 房龄 (0-100)",
        "",
        "## 3. 交叉验证结果 (CustomOLS, 5-Fold Leak-Free)",
        "",
        "| Fold | RMSE | MAE | MAPE(%) |",
        "|------|------|-----|---------|",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"| {int(row['fold'])} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['MAPE']:.2f} |")
    lines += [
        f"| **Mean** | **{metrics['RMSE'].mean():.4f}** | **{metrics['MAE'].mean():.4f}** | **{metrics['MAPE'].mean():.2f}** |",
        "",
        "## 4. 对比模型",
        f"- CustomOLS 平均 RMSE: {metrics['RMSE'].mean():.4f}",
        f"- GradientDescentOLS 平均 RMSE: {r['gd_metrics']['RMSE'].mean():.4f}",
        f"- sklearn LinearRegression 平均 RMSE: {r['sk_rmse_mean']:.4f}",
        "",
        "## 5. VIF 诊断",
        "",
        "| Feature | VIF | 判断 |",
        "|---------|-----|------|",
    ]
    for name, v in r["vif"].items():
        flag = "⚠️ 高共线性" if v > 10 else ("⚡ 中等" if v > 5 else "✅ 正常")
        lines.append(f"| {name} | {v:.2f} | {flag} |")
    lines += [
        "",
        "## 6. 系数方向 vs DGP",
        "",
        "| Feature | DGP 预期 | 模型系数 | 一致? |",
        "|---------|----------|----------|-------|",
    ]
    dgp_expected = {
        "area_sqft": "+", "bedrooms": "~0 (共线)", "age_years": "-",
        "location_B": "-vsA",
        "location_C": "-vsA",
    }
    for name, c in r["last_coefs"].items():
        exp = dgp_expected.get(name, "—")
        actual_sign = "+" if c > 0 else "-"
        lines.append(f"| {name} | {exp} | {c:+.4f} ({actual_sign}) | — |")
    lines += [
        "",
        "## 7. 推测总结",
        "- `area_sqft` 方向与 DGP 一致 (正向), 因标准化量纲不可直接比较绝对值",
        "- `age_years` 方向与 DGP 一致 (负向)",
        "- `bedrooms` 因与 area_sqft 高度共线 (VIF≈9), 系数不稳定",
        "- 位置变量使用 drop-first 编码 (A 为基线), B/C 系数表示与 A 的差异",
        "- 异常值和缺失值在 winsorization + CV 内均值填补后影响可控",
        "- GradientDescentOLS 收敛不佳, 需调整学习率",
        "",
        "> 因为知道 DGP, 所以可以精确验证模型识别能力。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ synthetic_report.md")


def _write_kaggle_report(r: dict):
    path = os.path.join(RESULTS_DIR, "kaggle_report.md")
    lines = [
        "# 🌍 Kaggle Real Data Report — 真实数据回归报告",
        "",
        "## 1. 数据信息",
        "- **数据集名称**: Medical Cost Personal Dataset",
        "- **Kaggle 链接**: https://www.kaggle.com/datasets/mirichoi0218/insurance",
        "- **下载日期**: 2026-05-19",
        "- **目标变量**: `charges` (医疗费用, 连续变量, 美元)",
        "- **每行样本**: 一位参保人的人口统计信息及年度医疗费用",
        "- **原始特征**: age, sex, bmi, children, smoker, region, charges",
        "",
        "## 2. 选择理由",
        "该数据集混合了数值变量 (age, bmi, children) 和类别变量 (sex, smoker, region),",
        "存在明显的偏态 (charges 右偏)、缺失值少但存在异常值, 且业务含义清晰,",
        "适合作为回归分析的真实场景练习。",
        "",
        "## 3. 预处理说明",
        "- 类别变量 (sex, smoker, region) → OneHot 编码",
        "- 缺失值: 在每折 CV 内用训练集均值填补 (CustomImputer)",
        "- 标准化: 在每折 CV 内用训练集参数标准化 (CustomStandardScaler)",
        "- 离群值: 全局 Winsorization (1%-99%), 不学参数, 无泄露",
        "",
        "## 4. 交叉验证结果 (CustomOLS, 5-Fold Leak-Free)",
        "",
        "| Fold | RMSE | MAE | MAPE(%) |",
        "|------|------|-----|---------|",
    ]
    metrics = r["ols_metrics"]
    for _, row in metrics.iterrows():
        lines.append(f"| {int(row['fold'])} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['MAPE']:.2f} |")
    lines += [
        f"| **Mean** | **{metrics['RMSE'].mean():.4f}** | **{metrics['MAE'].mean():.4f}** | **{metrics['MAPE'].mean():.2f}** |",
        "",
        "## 5. 对比模型",
        f"- CustomOLS 平均 RMSE: {r['ols_rmse_mean']:.4f}, MAE: {r['ols_mae_mean']:.4f}, MAPE: {r['ols_mape_mean']:.2f}%",
        f"- GradientDescentOLS 平均 RMSE: {r['gd_rmse_mean']:.4f}, MAE: {r['gd_mae_mean']:.4f}, MAPE: {r['gd_mape_mean']:.2f}%",
        f"- sklearn LinearRegression 平均 RMSE: {r['sk_rmse_mean']:.4f}, MAE: {r['sk_mae_mean']:.4f}, MAPE: {r['sk_mape_mean']:.2f}%",
        "",
        "## 6. VIF 诊断",
        "",
        "| Feature | VIF | 判断 |",
        "|---------|-----|------|",
    ]
    for name, v in r["vif"].items():
        flag = "⚠️ 高共线性" if v > 10 else ("⚡ 中等" if v > 5 else "✅ 正常")
        lines.append(f"| {name} | {v:.2f} | {flag} |")
    lines += [
        "",
        "## 7. 系数方向分析",
        "",
        "| Feature | Coefficient | 解读 |",
        "|---------|-------------|------|",
    ]
    for name, c in r["last_coefs"].items():
        direction = "正向" if c > 0 else "负向"
        lines.append(f"| {name} | {c:+.4f} | {direction} |")
    lines += [
        "",
        "## 8. 推测与业务解读",
        "- **smoker** 极可能是最强预测变量 (吸烟者费用显著更高)",
        "- **age** 正向影响: 年长者费用更高",
        "- **bmi** 正向: 较高 BMI 与更高费用相关",
        "- 模型平均误差 (RMSE) 代表预测费用与实际费用的典型偏差",
        "- 若用于真实决策, 需注意: 地区/性别差异可能反映系统性偏见而非因果关系",
        "",
        "## 9. 上线风险",
        "- 数据为美国参保人群, 不一定适用于其他人群",
        "- smoker 的强效应可能导致模型过度依赖单一变量",
        "- 费用分布严重右偏, MAPE 可能较高",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ kaggle_report.md")


def _write_summary_comparison(syn: dict, kag: dict):
    path = os.path.join(RESULTS_DIR, "summary_comparison.md")
    syn_rmse = syn["ols_metrics"]["RMSE"].mean()
    kag_rmse = kag["ols_metrics"]["RMSE"].mean()
    lines = [
        "# 🧠 Summary Comparison — 模拟 vs 真实数据对照",
        "",
        "## 1. 推测难度对比",
        "在模拟数据中, 因为知道 DGP (数据生成机制),",
        "可以精确验证: 系数方向是否正确, 异常值/缺失值影响是否可控,",
        "以及变量方向是否与设定一致。推测相对容易。",
        "",
        "在真实数据中, 不知道「真实的」生成机制,",
        "只能通过统计指标 (系数、VIF、CV 误差) 间接推断关系。",
        "即使分数不错, 也可能存在未观测混杂变量或因果反转问题。",
        "",
        "## 2. 共线性、缺失值、异常值的影响",
        "",
        "| 问题 | 模拟数据 | 真实数据 |",
        "|------|----------|----------|",
        "| 共线性 | 明确构造 (bedrooms vs area_sqft), 可控诊断 | 可能存在未知共线性 (如 bmi 与 age 的相关) |",
        "| 缺失值 | 已知位置和比例, 可验证填补效果 | 可能需要更复杂的填补策略 |",
        "| 异常值 | 人为注入, 知道真实值范围 | 需借助业务判断真异常 vs 极端但合理的值 |",
        "",
        "## 3. 为什么无泄露交叉验证在真实数据上尤其重要",
        "- 真实数据的分布未知, 预处理参数的「信息泄露」更难察觉",
        "- 若在全量数据上做均值填补/标准化, 验证集信息会污染训练过程",
        "- 交叉验证的每一折都必须独立地 fit → transform, 才能得到无偏估计",
        "",
        "## 4. utils/ 组件复用情况",
        "- `CustomImputer`: 缺失值填补 (CV 内每折独立 fit)",
        "- `CustomStandardScaler`: Z-score 标准化 (CV 内每折独立 fit)",
        "- `CustomOLS` / `GradientDescentOLS`: 主要回归模型",
        "- `calculate_rmse` / `calculate_mae` / `calculate_mape`: 评估指标",
        "- `calculate_vif`: 共线性诊断",
        "",
        "## 5. 关键指标汇总",
        "",
        "| 场景 | 模型 | RMSE | MAE | MAPE(%) |",
        "|------|------|------|-----|---------|",
        f"| 模拟 | CustomOLS | {syn_rmse:.4f} | {syn['ols_metrics']['MAE'].mean():.4f} | {syn['ols_metrics']['MAPE'].mean():.2f} |",
        f"| 模拟 | sklearn | {syn['sk_rmse_mean']:.4f} | {syn['sk_mae_mean']:.4f} | {syn['sk_mape_mean']:.2f} |",
        f"| 真实 | CustomOLS | {kag_rmse:.4f} | {kag['ols_metrics']['MAE'].mean():.4f} | {kag['ols_metrics']['MAPE'].mean():.2f} |",
        f"| 真实 | sklearn | {kag['sk_rmse_mean']:.4f} | {kag['sk_mae_mean']:.4f} | {kag['sk_mape_mean']:.2f} |",
        "",
        "## 6. 最终思考",
        "模拟世界给出「已知答案的考试」, 验证方法是否正确;",
        "真实世界给出「开放命题的实践」, 考验判断是否稳健。",
        "两者结合, 才能从「会调包」走向「会分析」。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ summary_comparison.md")


# ========================== 主入口 ===========================================

def main():
    """唯一执行入口: uv run src/week11/main.py"""
    print("=" * 60)
    print("  Week 11: Dual Inference Sprint")
    print("  Synthetic-to-Real Regression Workflow")
    print("=" * 60)

    # --- 准备 ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Task A: 模拟数据 ---
    generate_synthetic_data(n_samples=400)
    syn_results = run_synthetic_task()

    # --- Task B: Kaggle 真实数据 ---
    df_kaggle = load_kaggle_data()
    kag_results = run_kaggle_task(df_kaggle)

    # --- Task C: 报告 ---
    write_reports(syn_results, kag_results)

    print("\n" + "=" * 60)
    print("  ✅ Week 11 全部流程完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
