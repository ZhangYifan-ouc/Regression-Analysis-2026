"""Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow.

Single entry point:
    uv run week11/main.py

The script intentionally keeps all learned preprocessing inside each CV fold:
training fold -> fit imputer/winsorizer/scaler/encoder -> transform validation fold.
This avoids the data leakage problem emphasized in Week 10 and Week 11.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Make imports work when this file is executed as: uv run week11/main.py
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import add_intercept, calculate_vif, correlation_pairs, residual_summary
from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import AnalyticalOLS
from utils.transformers import RegressionPreprocessor

ROOT_DIR = PROJECT_DIR
WEEK11_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK11_DIR / "data"
RESULTS_DIR = WEEK11_DIR / "results"
SYNTHETIC_PATH = DATA_DIR / "synthetic_regression.csv"
KAGGLE_PATH = DATA_DIR / "kaggle_gapminder.csv"


def ensure_directories() -> None:
    """Create data directory and refresh results directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    """Small markdown table writer without depending on tabulate."""
    if df is None or df.empty:
        return "_No rows._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "")
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def generate_synthetic_data(output_path: Path = SYNTHETIC_PATH, seed: int = 2026, n: int = 600) -> pd.DataFrame:
    """Generate a marketing-like regression dataset with a known DGP."""
    rng = np.random.default_rng(seed)

    tv_budget = rng.normal(loc=85, scale=24, size=n).clip(15, 160)
    # Deliberate collinearity: radio is generated from TV plus small noise.
    radio_budget = (0.82 * tv_budget + rng.normal(loc=0, scale=6, size=n)).clip(5, 150)
    search_clicks = rng.lognormal(mean=10.1, sigma=0.45, size=n)
    discount_rate = rng.beta(a=2.2, b=8.5, size=n) * 0.45
    competitor_price_index = rng.normal(loc=100, scale=9, size=n).clip(70, 130)
    brand_index = rng.normal(loc=55, scale=12, size=n).clip(15, 95)

    channel = rng.choice(["offline", "search", "social", "partner"], size=n, p=[0.28, 0.35, 0.25, 0.12])
    season = rng.choice(["normal", "holiday", "back_to_school"], size=n, p=[0.58, 0.25, 0.17])

    channel_effect = {"offline": 0, "search": 2400, "social": 1700, "partner": 1100}
    season_effect = {"normal": 0, "holiday": 3600, "back_to_school": 1900}
    noise = rng.normal(loc=0, scale=4200, size=n)

    q4_sales = (
        18000
        + 130 * tv_budget
        + 55 * radio_budget
        + 0.20 * search_clicks
        - 21000 * discount_rate
        + 90 * competitor_price_index
        + 160 * brand_index
        + np.vectorize(channel_effect.get)(channel)
        + np.vectorize(season_effect.get)(season)
        + noise
    )

    df = pd.DataFrame(
        {
            "tv_budget": tv_budget,
            "radio_budget": radio_budget,
            "search_clicks": search_clicks,
            "discount_rate": discount_rate,
            "competitor_price_index": competitor_price_index,
            "brand_index": brand_index,
            "channel": channel,
            "season": season,
            "q4_sales": q4_sales,
        }
    )

    # Inject real-world issues: missing values, outliers, scale differences and collinearity.
    for col, frac in {"tv_budget": 0.045, "search_clicks": 0.035, "channel": 0.03}.items():
        idx = rng.choice(n, size=max(1, int(n * frac)), replace=False)
        df.loc[idx, col] = np.nan

    outlier_idx = rng.choice(n, size=int(n * 0.025), replace=False)
    df.loc[outlier_idx, "search_clicks"] *= rng.uniform(3.5, 6.0, size=len(outlier_idx))
    sales_outlier_idx = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[sales_outlier_idx, "q4_sales"] += rng.normal(25000, 6500, size=len(sales_outlier_idx))

    df.to_csv(output_path, index=False)
    return df


def load_synthetic_data() -> pd.DataFrame:
    if not SYNTHETIC_PATH.exists():
        return generate_synthetic_data(SYNTHETIC_PATH)
    return pd.read_csv(SYNTHETIC_PATH)


def load_kaggle_data(path: Path = KAGGLE_PATH) -> pd.DataFrame:
    """Load the selected Kaggle real-world regression dataset."""
    if not path.exists():
        raise FileNotFoundError(
            "Missing week11/data/kaggle_gapminder.csv. "
            "The submitted package should include this Kaggle data file."
        )
    df = pd.read_csv(path)
    required = {"country", "continent", "year", "lifeExp", "pop", "gdpPercap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kaggle file is missing required columns: {sorted(missing)}")
    return df


def clean_kaggle_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and feature engineering for Gapminder life expectancy data."""
    df = raw.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    for col in ["year", "lifeExp", "pop", "gdpPercap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["lifeExp"] > 0) & (df["pop"] > 0) & (df["gdpPercap"] > 0)].copy()
    df["year_centered"] = df["year"] - df["year"].min()
    df["log_population"] = np.log(df["pop"])
    df["log_gdp_per_capita"] = np.log(df["gdpPercap"])
    # Keep country for business explanation, but do not use it as a model feature
    # because 142 country dummies would make interpretation unstable for this homework.
    return df.reset_index(drop=True)


def cross_validate_custom_workflow(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    include_sklearn_baseline: bool = True,
) -> dict[str, Any]:
    """Run leakage-free 5-fold CV with custom preprocessing and custom OLS."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    custom_fold_rows: list[dict[str, float | int]] = []
    baseline_fold_rows: list[dict[str, float | int]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_id, (train_idx, val_idx) in enumerate(kfold.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Fit learned preprocessing parameters only on the training fold.
        preprocessor = RegressionPreprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            impute_strategy="median",
            winsor_limits=(0.01, 0.99),
            drop_first=True,
        )
        X_train = preprocessor.fit_transform(train_df)
        X_val = preprocessor.transform(val_df)
        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)

        custom_model = AnalyticalOLS().fit(add_intercept(X_train), y_train)
        pred_custom = custom_model.predict(add_intercept(X_val))
        custom_fold_rows.append(
            {
                "fold": fold_id,
                "RMSE": calculate_rmse(y_val, pred_custom),
                "MAE": calculate_mae(y_val, pred_custom),
                "MAPE": calculate_mape(y_val, pred_custom),
                "R2": custom_model.score(add_intercept(X_val), y_val),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "row_id": val_idx,
                "fold": fold_id,
                "y_true": y_val,
                "custom_pred": pred_custom,
            }
        )

        if include_sklearn_baseline:
            baseline = Ridge(alpha=1.0, fit_intercept=True)
            baseline.fit(X_train, y_train)
            pred_baseline = baseline.predict(X_val)
            baseline_fold_rows.append(
                {
                    "fold": fold_id,
                    "RMSE": calculate_rmse(y_val, pred_baseline),
                    "MAE": calculate_mae(y_val, pred_baseline),
                    "MAPE": calculate_mape(y_val, pred_baseline),
                    "R2": baseline.score(X_val, y_val),
                }
            )
            fold_predictions["sklearn_ridge_pred"] = pred_baseline

        prediction_rows.append(fold_predictions)

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("row_id")
    custom_folds = pd.DataFrame(custom_fold_rows)
    baseline_folds = pd.DataFrame(baseline_fold_rows) if baseline_fold_rows else pd.DataFrame()

    # Full-data fit is used only for interpretation/diagnostics after CV, not for evaluation.
    full_preprocessor = RegressionPreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        impute_strategy="median",
        winsor_limits=(0.01, 0.99),
        drop_first=True,
    )
    X_full = full_preprocessor.fit_transform(df)
    y_full = df[target].to_numpy(dtype=float)
    full_model = AnalyticalOLS().fit(add_intercept(X_full), y_full)
    full_pred = full_model.predict(add_intercept(X_full))
    feature_names = ["intercept"] + (full_preprocessor.feature_names_ or [])
    coef_table = pd.DataFrame({"feature": feature_names, "coefficient": full_model.coef_})
    vif_table = calculate_vif(X_full, full_preprocessor.feature_names_ or [])

    return {
        "custom_folds": custom_folds,
        "baseline_folds": baseline_folds,
        "predictions": predictions,
        "coef_table": coef_table,
        "vif_table": vif_table,
        "residual_summary": residual_summary(y_full, full_pred),
        "feature_names": full_preprocessor.feature_names_,
    }


def summarize_cv(fold_metrics: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return mean and std of CV metrics."""
    metric_cols = [c for c in ["RMSE", "MAE", "MAPE", "R2"] if c in fold_metrics.columns]
    rows = []
    for metric in metric_cols:
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "mean": float(fold_metrics[metric].mean()),
                "std": float(fold_metrics[metric].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def make_actual_vs_pred_plot(predictions: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(6.5, 5))
    plt.scatter(predictions["y_true"], predictions["custom_pred"], alpha=0.65)
    lo = float(min(predictions["y_true"].min(), predictions["custom_pred"].min()))
    hi = float(max(predictions["y_true"].max(), predictions["custom_pred"].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_residual_plot(predictions: pd.DataFrame, title: str, output_path: Path) -> None:
    residuals = predictions["y_true"] - predictions["custom_pred"]
    plt.figure(figsize=(6.5, 5))
    plt.scatter(predictions["custom_pred"], residuals, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def prepare_synthetic_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with valid target and return modeling frame."""
    out = df.copy()
    out["q4_sales"] = pd.to_numeric(out["q4_sales"], errors="coerce")
    out = out[out["q4_sales"].notna()].reset_index(drop=True)
    return out


def run_synthetic_task() -> dict[str, Any]:
    synthetic_raw = generate_synthetic_data(SYNTHETIC_PATH)
    synthetic = prepare_synthetic_for_model(synthetic_raw)
    numeric_features = [
        "tv_budget",
        "radio_budget",
        "search_clicks",
        "discount_rate",
        "competitor_price_index",
        "brand_index",
    ]
    categorical_features = ["channel", "season"]
    result = cross_validate_custom_workflow(
        synthetic,
        target="q4_sales",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_sklearn_baseline=True,
    )
    result["data"] = synthetic
    result["numeric_features"] = numeric_features
    result["categorical_features"] = categorical_features
    make_actual_vs_pred_plot(
        result["predictions"],
        "Synthetic data: actual vs predicted sales",
        RESULTS_DIR / "synthetic_actual_vs_pred.png",
    )
    make_residual_plot(
        result["predictions"],
        "Synthetic data: residual plot",
        RESULTS_DIR / "synthetic_residuals.png",
    )
    write_synthetic_report(result)
    return result


def run_kaggle_task() -> dict[str, Any]:
    raw = load_kaggle_data(KAGGLE_PATH)
    kaggle = clean_kaggle_data(raw)
    numeric_features = ["year_centered", "log_gdp_per_capita", "log_population"]
    categorical_features = ["continent"]
    result = cross_validate_custom_workflow(
        kaggle,
        target="lifeExp",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_sklearn_baseline=True,
    )
    result["raw_data"] = raw
    result["data"] = kaggle
    result["numeric_features"] = numeric_features
    result["categorical_features"] = categorical_features
    make_actual_vs_pred_plot(
        result["predictions"],
        "Kaggle Gapminder: actual vs predicted life expectancy",
        RESULTS_DIR / "kaggle_actual_vs_pred.png",
    )
    make_residual_plot(
        result["predictions"],
        "Kaggle Gapminder: residual plot",
        RESULTS_DIR / "kaggle_residuals.png",
    )
    write_kaggle_report(result)
    return result


def data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].describe().T.reset_index().rename(columns={"index": "feature"})[
        ["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]


def coefficient_direction_table(coefs: pd.DataFrame, expected: dict[str, str]) -> pd.DataFrame:
    rows = []
    for feature, direction in expected.items():
        match = coefs[coefs["feature"] == feature]
        if match.empty:
            continue
        coef = float(match["coefficient"].iloc[0])
        actual_direction = "正向" if coef > 0 else "负向" if coef < 0 else "接近 0"
        rows.append(
            {
                "feature": feature,
                "expected": direction,
                "estimated_coef": coef,
                "estimated_direction": actual_direction,
                "consistent": "是" if direction == actual_direction else "否/需解释",
            }
        )
    return pd.DataFrame(rows)


def write_synthetic_report(result: dict[str, Any]) -> None:
    df = result["data"]
    custom_summary = summarize_cv(result["custom_folds"], "Custom OLS workflow")
    baseline_summary = summarize_cv(result["baseline_folds"], "sklearn Ridge baseline")
    summary = pd.concat([custom_summary, baseline_summary], ignore_index=True)
    corr = correlation_pairs(df[result["numeric_features"] + ["q4_sales"]], threshold=0.70)
    expected = {
        "tv_budget": "正向",
        "radio_budget": "正向",
        "search_clicks": "正向",
        "discount_rate": "负向",
        "competitor_price_index": "正向",
        "brand_index": "正向",
        "channel__search": "正向",
        "channel__social": "正向",
        "season__holiday": "正向",
    }
    direction_table = coefficient_direction_table(result["coef_table"], expected)

    text = f"""# Week 11 Task A：模拟数据回归报告

## 1. 数据生成机制（DGP）

本任务构造了一个 Q4 营销投放与销售额场景。每一行代表一个产品/区域/渠道组合在 Q4 的投放记录，目标变量为 `q4_sales`。

显式设定的目标生成公式为：

```text
q4_sales = 18000
         + 130 * tv_budget
         + 55 * radio_budget
         + 0.20 * search_clicks
         - 21000 * discount_rate
         + 90 * competitor_price_index
         + 160 * brand_index
         + channel_effect
         + season_effect
         + random_noise
```

其中：

- `tv_budget`、`radio_budget`、`search_clicks`、`competitor_price_index`、`brand_index` 应正向影响销售额；
- `discount_rate` 的系数被设为负数，表示过度折扣可能压低收入质量；
- `channel=search/social/partner` 相对于 `offline` 有正向渠道效果；
- `season=holiday/back_to_school` 相对于 `normal` 有正向季节效果；
- 我故意构造了 `radio_budget = 0.82 * tv_budget + noise`，因此 `tv_budget` 与 `radio_budget` 应该高度相关。

主动加入的真实世界问题包括：缺失值、异常值、明显量纲差异、共线性。生成数据已保存到 `week11/data/synthetic_regression.csv`。

## 2. 数据概览

- 样本量：{len(df)}
- 特征数：{len(result['numeric_features']) + len(result['categorical_features'])}
- 目标变量：`q4_sales`

### 字段质量检查

{df_to_markdown(data_quality_table(df), float_digits=4)}

### 数值变量描述性统计

{df_to_markdown(numeric_summary(df, result['numeric_features'] + ['q4_sales']), float_digits=2)}

## 3. 无泄露建模流程

5 折交叉验证中，每一折都按以下顺序处理：

1. 只在训练折上 `fit` 自定义 `RegressionPreprocessor`；
2. 训练折学习中位数填补、winsorization 分位点、标准化均值/标准差、类别编码结构；
3. 验证折只调用 `transform()`，不重新学习任何参数；
4. 主模型使用 `utils.models.AnalyticalOLS`；
5. 指标使用 `utils.metrics.calculate_rmse / calculate_mae / calculate_mape`；
6. VIF 使用 `utils.diagnostics.calculate_vif`。

## 4. 5 折交叉验证结果

### 每折结果（自定义 OLS 主流程）

{df_to_markdown(result['custom_folds'], float_digits=3)}

### 指标均值与标准差

{df_to_markdown(summary, float_digits=3)}

说明：`sklearn Ridge baseline` 仅作对照，预处理仍使用本作业自己的 transformer；主流程结论以 `Custom OLS workflow` 为准。

## 5. 共线性诊断

### 高相关变量对

{df_to_markdown(corr, max_rows=10, float_digits=3)}

### VIF 前 12 项

{df_to_markdown(result['vif_table'], max_rows=12, float_digits=3)}

`tv_budget` 与 `radio_budget` 的 VIF 明显较高，符合我在 DGP 中主动构造共线性的预期。这说明系数解释时不能只看单个变量的正负和大小，而要把这一组高度相关投放变量作为整体解释。

## 6. 系数方向与 DGP 是否一致

{df_to_markdown(direction_table, float_digits=3)}

整体上，主要变量方向与 DGP 基本一致。`tv_budget` 与 `radio_budget` 可能出现单个系数不稳定，是因为二者被故意设为高度相关；在线性回归里，模型很难稳定地区分“电视投放”和“广播投放”的独立边际贡献。

## 7. 推测结论

- 模拟数据中，因为真实 DGP 已知，所以判断变量方向是否正确相对容易。
- 影响最稳定的是 `search_clicks`、`brand_index`、季节变量和渠道变量，它们与目标的关系不完全依赖共线变量组。
- `tv_budget` 与 `radio_budget` 本来就难以稳定识别，因为二者高度相关，VIF 也验证了这一点。
- RMSE 比 MAE 更容易受到我主动加入的销售额异常值影响，因此报告中同时保留 RMSE、MAE、MAPE 三个指标。

## 8. 图形

- `synthetic_actual_vs_pred.png`
- `synthetic_residuals.png`
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(text, encoding="utf-8")


def write_kaggle_report(result: dict[str, Any]) -> None:
    df = result["data"]
    raw = result["raw_data"]
    custom_summary = summarize_cv(result["custom_folds"], "Custom OLS workflow")
    baseline_summary = summarize_cv(result["baseline_folds"], "sklearn Ridge baseline")
    summary = pd.concat([custom_summary, baseline_summary], ignore_index=True)
    corr = correlation_pairs(df[["year_centered", "log_gdp_per_capita", "log_population", "lifeExp"]], threshold=0.70)
    expected = {
        "year_centered": "正向",
        "log_gdp_per_capita": "正向",
        "log_population": "正向",
        "continent__Americas": "正向",
        "continent__Europe": "正向",
        "continent__Oceania": "正向",
    }
    direction_table = coefficient_direction_table(result["coef_table"], expected)

    text = f"""# Week 11 Task B：Kaggle 真实数据回归报告

## 1. Kaggle 数据说明

- 数据集名称：Gapminder: Countries Over Time
- Kaggle 页面链接：https://www.kaggle.com/datasets/shraddha4ever20/gapminder-countries-over-time
- 下载/整理日期：2026-05-24
- 使用文件：`week11/data/kaggle_gapminder.csv`
- 预测目标：`lifeExp`，即出生时预期寿命，连续变量，适合回归问题。
- 每一行样本代表：某个国家在某个年份的国家层面观测，包括洲别、人口、GDP per capita 和预期寿命。

我选择这份数据，是因为它有真实业务/公共政策含义：预期寿命会受到经济发展水平、人口规模、时间趋势和地区差异共同影响。相比几乎只需直接套模板的数据，这份数据需要处理强偏态数值变量、高基数国家字段、时间趋势和地区类别解释问题。

## 2. 原始数据概览

- 原始行数：{len(raw)}
- 清洗后行数：{len(df)}
- 原始字段：{', '.join(raw.columns)}

### 字段质量检查

{df_to_markdown(data_quality_table(raw), float_digits=4)}

### 建模后数值变量描述性统计

{df_to_markdown(numeric_summary(df, ['year_centered', 'log_gdp_per_capita', 'log_population', 'lifeExp']), float_digits=3)}

## 3. 清洗与特征处理

本任务做了以下处理：

1. 删除完全重复行；
2. 将 `year`、`lifeExp`、`pop`、`gdpPercap` 转为数值；
3. 删除目标或关键数值字段非正的记录；
4. 构造 `year_centered = year - min(year)`，避免年份原值过大；
5. 对 `pop` 和 `gdpPercap` 做对数变换，得到 `log_population` 和 `log_gdp_per_capita`，缓解强偏态和极端值影响；
6. 保留 `continent` 做类别变量；
7. 不把 `country` 作为模型特征，因为 142 个国家哑变量会让作业中的线性解释过度复杂，且容易引入国家固定效应主导问题。

在每一个 CV 训练折中，自定义 `RegressionPreprocessor` 会重新学习：中位数填补、winsorization 分位点、标准化参数和类别编码结构。验证折只 `transform`，没有泄露。

## 4. 5 折交叉验证结果

### 每折结果（自定义 OLS 主流程）

{df_to_markdown(result['custom_folds'], float_digits=3)}

### 指标均值与标准差

{df_to_markdown(summary, float_digits=3)}

`sklearn Ridge baseline` 只是对照组，主要分析仍来自自定义预处理 + 自定义 OLS + 自定义 metrics 的主流程。

## 5. 共线性与诊断

### 高相关变量对

{df_to_markdown(corr, max_rows=10, float_digits=3)}

### VIF 前 12 项

{df_to_markdown(result['vif_table'], max_rows=12, float_digits=3)}

### 残差摘要

{df_to_markdown(pd.DataFrame([result['residual_summary']]), float_digits=3)}

`year_centered` 和 `log_gdp_per_capita` 往往会同时与 `lifeExp` 正相关，因为全球经济发展和医疗改善都随时间推进。VIF 用来提醒我们：如果某些变量互相解释能力过强，单个系数的业务解释就要谨慎。

## 6. 系数方向与真实数据推测

{df_to_markdown(direction_table, float_digits=3)}

较稳定的变量通常是：

- `year_centered`：时间推进通常对应公共卫生、医疗和教育改善；
- `log_gdp_per_capita`：经济水平更高通常对应更高预期寿命；
- `continent`：地区差异吸收了很多制度、地理和历史因素。

不稳定或需要谨慎解释的变量包括：

- `log_population`：人口规模本身不必然提高或降低寿命，它更像国家规模控制变量；
- `continent` 的哑变量：它不是直接因果变量，而是很多未观测因素的综合代理；
- `country` 被删除，因为它会让模型更多记住国家身份，而不是学习可解释的经济/人口关系。

## 7. 业务解释与上线风险

平均误差可以理解为模型对国家-年份预期寿命的平均预测偏差。即使 RMSE/MAE 看起来可接受，这个模型也不能直接用于政策因果判断，因为：

1. 数据是国家层面聚合数据，不能推断个人层面的寿命；
2. `continent` 代表的是复杂地区差异，不是可干预变量；
3. 时间趋势、GDP、医疗水平、战争、疫情等因素互相交织；
4. 线性模型可能无法捕捉预期寿命的上限效应和非线性变化。

如果上线，我最担心的是把相关性误读为因果性，以及在极端国家/年份上预测偏差过大。

## 8. 图形

- `kaggle_actual_vs_pred.png`
- `kaggle_residuals.png`
"""
    (RESULTS_DIR / "kaggle_report.md").write_text(text, encoding="utf-8")


def write_summary_comparison(synthetic_result: dict[str, Any], kaggle_result: dict[str, Any]) -> None:
    synthetic_summary = summarize_cv(synthetic_result["custom_folds"], "Synthetic / Custom OLS")
    kaggle_summary = summarize_cv(kaggle_result["custom_folds"], "Kaggle / Custom OLS")
    combined = pd.concat([synthetic_summary, kaggle_summary], ignore_index=True)

    text = f"""# Week 11 Task C：模拟数据与 Kaggle 真实数据对照总结

## 1. 指标对照

{df_to_markdown(combined, float_digits=3)}

## 2. 为什么模拟数据中的“推测”更容易？

模拟数据的 DGP 是我自己写出来的，因此我事先知道哪些变量应该正向影响目标、哪些变量应该负向影响目标，也知道 `tv_budget` 和 `radio_budget` 被故意设置成高度相关。这样一来，模型结果可以直接和真实公式对照：如果系数方向不一致，优先检查噪声、异常值、共线性或预处理。

## 3. 为什么真实数据即使分数还可以，解释也更困难？

Kaggle Gapminder 数据中，`lifeExp` 与 GDP、年份、地区之间存在明显相关，但这些相关并不等于因果。比如洲别哑变量可能同时代表医疗体系、气候、历史、制度、战争风险等大量未观测因素。模型分数说明预测还可以，但不能说明“提高某个变量一定导致寿命变化”。真实数据的解释困难主要来自遗漏变量、聚合数据、时间趋势和变量代理含义不清晰。

## 4. 共线性、缺失值、异常值在两类数据上的影响

- 模拟数据：共线性是主动构造的，所以可以清楚地看到 `tv_budget` 与 `radio_budget` 的 VIF 风险；缺失值和异常值也是可控注入的，主要用于检验自己的清洗流程。
- 真实数据：共线性和异常值来自真实世界结构，例如 GDP 和年份可能共同反映长期发展趋势，人口和 GDP 的分布极端偏态。我们不知道完整 DGP，因此只能用 VIF、残差图和业务常识谨慎判断。

## 5. 为什么无泄露交叉验证在真实数据上尤其重要？

真实数据中的分布差异更复杂。如果在 CV 前对全量数据做均值填补、标准化或分位数截尾，就等于让验证折的信息提前进入训练过程，模型评估会过于乐观。Week11 的主流程在每一折中重新 `fit` 预处理器，然后对验证折只 `transform`，因此验证结果更接近未来新数据场景。

## 6. 自己维护的 utils 组件节省了哪些重复劳动？

本周复用了以下组件：

- `utils.models.AnalyticalOLS`：统一训练和预测接口；
- `utils.metrics.calculate_rmse / calculate_mae / calculate_mape`：统一指标计算；
- `utils.transformers.RegressionPreprocessor`：把缺失值填补、winsorization、标准化、one-hot 编码组织到同一个可复用对象；
- `utils.diagnostics.calculate_vif / correlation_pairs / residual_summary`：统一诊断输出。

因为这些工具已经封装好，模拟数据和 Kaggle 数据可以共用同一套 CV 函数，不需要复制粘贴两份清洗和评估代码。

## 7. 答辩准备要点

1. `main()` 的流程：清空结果目录 → 生成模拟数据 → 跑模拟任务 → 读取 Kaggle 数据 → 跑真实数据任务 → 写对照总结。
2. 无泄露关键：每一折训练集 `fit` 预处理器，验证集只 `transform`。
3. 主模型：自定义 `AnalyticalOLS`，不是 sklearn 一把梭。
4. baseline：`sklearn Ridge` 只是对照组，且仍使用自定义预处理后的特征。
5. 核心函数可解释：`RegressionPreprocessor.fit()` 学习填补值、截尾分位点、标准化参数和类别集合；`transform()` 只复用这些参数。
"""
    (RESULTS_DIR / "summary_comparison.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    synthetic_result = run_synthetic_task()
    kaggle_result = run_kaggle_task()
    write_summary_comparison(synthetic_result, kaggle_result)
    print("Week 11 workflow completed.")
    print(f"Synthetic data: {SYNTHETIC_PATH.relative_to(ROOT_DIR)}")
    print(f"Kaggle data: {KAGGLE_PATH.relative_to(ROOT_DIR)}")
    print(f"Reports: {RESULTS_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
