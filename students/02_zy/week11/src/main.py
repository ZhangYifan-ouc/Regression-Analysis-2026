import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WEEK11_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WEEK11_DIR / "src"
DATA_DIR = WEEK11_DIR / "data"
RESULTS_DIR = WEEK11_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import calculate_vif, top_vif
from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import CustomOLS
from utils.transformers import CustomMeanImputer, CustomStandardScaler, Winsorizer


RANDOM_SEED = 42
N_FOLDS = 5


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def format_number(value: float, digits: int = 4) -> str:
    if value == float("inf"):
        return "inf"

    return f"{value:.{digits}f}"


def make_metrics_table(summary: dict) -> str:
    lines = [
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| RMSE | {summary['RMSE']:.4f} |",
        f"| MAE | {summary['MAE']:.4f} |",
        f"| MAPE (%) | {summary['MAPE']:.4f} |",
    ]

    return "\n".join(lines)


def make_fold_table(fold_metrics: list[dict]) -> str:
    lines = [
        "| 折数 | RMSE | MAE | MAPE (%) |",
        "|---:|---:|---:|---:|",
    ]

    for item in fold_metrics:
        lines.append(
            f"| {item['fold']} | "
            f"{item['RMSE']:.4f} | "
            f"{item['MAE']:.4f} | "
            f"{item['MAPE']:.4f} |"
        )

    return "\n".join(lines)


def make_vif_table(vif_results: list[dict], n: int = 10) -> str:
    lines = [
        "| 特征 | VIF |",
        "|---|---:|",
    ]

    for item in top_vif(vif_results, n=n):
        lines.append(f"| {item['feature']} | {format_number(item['vif'])} |")

    return "\n".join(lines)


def make_coef_table(coef_results: list[dict], n: int = 12) -> str:
    sorted_results = sorted(
        coef_results,
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )[:n]

    lines = [
        "| 特征 | 系数 | 方向 |",
        "|---|---:|---|",
    ]

    for item in sorted_results:
        coef = item["coefficient"]
        direction = "正向" if coef > 0 else "负向" if coef < 0 else "接近 0"
        lines.append(f"| {item['feature']} | {coef:.4f} | {direction} |")

    return "\n".join(lines)


def parse_money(value) -> float:
    cleaned = re.sub(r"[^0-9.]", "", str(value))

    if cleaned == "":
        return np.nan

    return float(cleaned)


def parse_mileage(value) -> float:
    cleaned = re.sub(r"[^0-9.]", "", str(value))

    if cleaned == "":
        return np.nan

    return float(cleaned)


def simplify_transmission(value) -> str:
    text = str(value).lower()

    if "manual" in text or "m/t" in text:
        return "Manual"

    if "automatic" in text or "a/t" in text or "dual" in text:
        return "Automatic"

    return "Other"


def generate_synthetic_data(output_path: Path, n_samples: int = 500) -> pd.DataFrame:
    """
    Generate a business-like synthetic regression dataset.

    Scenario:
    We predict sales from advertising budget, store traffic, competitor price,
    and region.

    DGP:
    sales = 200
            + 2.5 * tv_budget
            + 1.2 * online_budget
            + 0.15 * store_visits
            - 3.0 * competitor_price
            + region_effect
            + noise

    online_budget is constructed from tv_budget, so they are highly correlated.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    tv_budget = rng.normal(80, 20, n_samples)
    online_budget = 0.85 * tv_budget + rng.normal(0, 5, n_samples)
    store_visits = rng.normal(1000, 220, n_samples)
    competitor_price = rng.normal(60, 12, n_samples)

    regions = rng.choice(
        ["East", "West", "North", "South"],
        size=n_samples,
        p=[0.30, 0.25, 0.25, 0.20],
    )

    region_effect_map = {
        "East": 40,
        "West": 20,
        "North": -10,
        "South": -25,
    }

    region_effect = np.array([region_effect_map[item] for item in regions])
    noise = rng.normal(0, 45, n_samples)

    sales = (
        200
        + 2.5 * tv_budget
        + 1.2 * online_budget
        + 0.15 * store_visits
        - 3.0 * competitor_price
        + region_effect
        + noise
    )

    df = pd.DataFrame(
        {
            "tv_budget": tv_budget,
            "online_budget": online_budget,
            "store_visits": store_visits,
            "competitor_price": competitor_price,
            "region": regions,
            "sales": sales,
        }
    )

    # Add missing values.
    missing_tv = rng.choice(df.index, size=int(0.06 * n_samples), replace=False)
    missing_competitor = rng.choice(df.index, size=int(0.05 * n_samples), replace=False)

    df.loc[missing_tv, "tv_budget"] = np.nan
    df.loc[missing_competitor, "competitor_price"] = np.nan

    # Add outliers.
    outlier_online = rng.choice(df.index, size=int(0.02 * n_samples), replace=False)
    outlier_visits = rng.choice(df.index, size=int(0.02 * n_samples), replace=False)

    df.loc[outlier_online, "online_budget"] = df.loc[outlier_online, "online_budget"] * 4
    df.loc[outlier_visits, "store_visits"] = df.loc[outlier_visits, "store_visits"] * 3

    df.to_csv(output_path, index=False)

    return df


def clean_kaggle_data(input_path: Path) -> pd.DataFrame:
    """
    Clean used car data.

    Every row represents one used car.
    Target variable: price.
    """
    df = pd.read_csv(input_path)

    df = df.copy()

    df["price"] = df["price"].apply(parse_money)
    df["milage"] = df["milage"].apply(parse_mileage)

    df = df.dropna(subset=["price"])

    # Business-rule filtering:
    # remove extreme exotic / collector car prices outside normal used-car market.
    df = df[(df["price"] >= 1000) & (df["price"] <= 500000)].copy()

    df["vehicle_age"] = 2024 - df["model_year"]
    df["vehicle_age"] = df["vehicle_age"].clip(lower=0)

    df["fuel_type"] = df["fuel_type"].replace(
        {
            "–": np.nan,
            "not supported": np.nan,
        }
    )

    df["transmission_simple"] = df["transmission"].apply(simplify_transmission)

    df["accident"] = df["accident"].fillna("Unknown")
    df["clean_title"] = df["clean_title"].fillna("Unknown")
    df["fuel_type"] = df["fuel_type"].fillna("Unknown")
    df["brand"] = df["brand"].fillna("Unknown")

    keep_cols = [
        "price",
        "vehicle_age",
        "milage",
        "brand",
        "fuel_type",
        "transmission_simple",
        "accident",
        "clean_title",
    ]

    return df[keep_cols].copy()


def fit_category_maps(train_df: pd.DataFrame, categorical_cols: list[str], max_levels: int = 12):
    """
    Fit category maps on training data only.

    Low-frequency categories are grouped into 'Other'.
    Missing categories are grouped into 'Missing'.
    """
    category_maps = {}

    for col in categorical_cols:
        series = train_df[col].fillna("Missing").astype(str)
        top_categories = series.value_counts().head(max_levels).index.tolist()

        category_maps[col] = top_categories

    return category_maps


def apply_category_maps(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    df = df.copy()

    for col, top_categories in category_maps.items():
        values = df[col].fillna("Missing").astype(str)

        df[col] = np.where(values.isin(top_categories), values, "Other")

    return df


def build_design_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    category_maps: dict,
    train_columns: list[str] | None = None,
):
    """
    Build numeric + one-hot encoded design matrix.

    If train_columns is provided, align validation columns to training columns.
    """
    df = apply_category_maps(df, category_maps)

    numeric_part = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if categorical_cols:
        categorical_part = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=float)
        X_df = pd.concat([numeric_part, categorical_part], axis=1)
    else:
        X_df = numeric_part

    if train_columns is not None:
        X_df = X_df.reindex(columns=train_columns, fill_value=0.0)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)

    return X, feature_names


def make_folds(n_samples: int, n_folds: int = N_FOLDS):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n_samples)

    rng.shuffle(indices)

    return np.array_split(indices, n_folds)


def preprocess_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    """
    Leakage-free fold preprocessing.

    Fit category maps, winsorizer, imputer, and scaler on training data only.
    Validation data only uses transform().
    """
    category_maps = fit_category_maps(train_df, categorical_cols)

    X_train_raw, feature_names = build_design_matrix(
        train_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_maps=category_maps,
        train_columns=None,
    )

    X_val_raw, _ = build_design_matrix(
        val_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_maps=category_maps,
        train_columns=feature_names,
    )

    winsorizer = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
    imputer = CustomMeanImputer()
    scaler = CustomStandardScaler()

    X_train = winsorizer.fit_transform(X_train_raw)
    X_val = winsorizer.transform(X_val_raw)

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    y_train = train_df[target_col].to_numpy(dtype=float)
    y_val = val_df[target_col].to_numpy(dtype=float)

    return X_train, X_val, y_train, y_val, feature_names


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def run_leakage_free_cv(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    positive_target: bool = False,
):
    """
    Run leakage-free 5-fold CV using our own utils components.
    """
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    folds = make_folds(len(df), n_folds=N_FOLDS)

    fold_metrics = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(df)), val_idx)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        X_train, X_val, y_train, y_val, _ = preprocess_fold(
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        model = CustomOLS()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if positive_target:
            y_pred = np.clip(y_pred, 1, None)

        metrics = evaluate_predictions(y_val, y_pred)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    summary = {
        "RMSE": float(np.mean([item["RMSE"] for item in fold_metrics])),
        "MAE": float(np.mean([item["MAE"] for item in fold_metrics])),
        "MAPE": float(np.mean([item["MAPE"] for item in fold_metrics])),
    }

    return summary, fold_metrics


def fit_full_model(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    """
    Fit a final model on the full dataset for coefficient and VIF interpretation.
    This is not used for validation scores.
    """
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    category_maps = fit_category_maps(df, categorical_cols)

    X_raw, feature_names = build_design_matrix(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_maps=category_maps,
        train_columns=None,
    )

    winsorizer = Winsorizer()
    imputer = CustomMeanImputer()
    scaler = CustomStandardScaler()

    X = winsorizer.fit_transform(X_raw)
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    y = df[target_col].to_numpy(dtype=float)

    model = CustomOLS()
    model.fit(X, y)

    coef_results = []

    for name, coef in zip(feature_names, model.coef_[1:]):
        coef_results.append(
            {
                "feature": name,
                "coefficient": float(coef),
            }
        )

    vif_results = calculate_vif(X, feature_names=feature_names)

    return coef_results, vif_results


def run_synthetic_task() -> dict:
    synthetic_path = DATA_DIR / "synthetic_regression.csv"

    synthetic_df = generate_synthetic_data(synthetic_path)

    target_col = "sales"
    numeric_cols = [
        "tv_budget",
        "online_budget",
        "store_visits",
        "competitor_price",
    ]
    categorical_cols = ["region"]

    summary, fold_metrics = run_leakage_free_cv(
        df=synthetic_df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        positive_target=False,
    )

    coef_results, vif_results = fit_full_model(
        df=synthetic_df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    write_synthetic_report(
        summary=summary,
        fold_metrics=fold_metrics,
        coef_results=coef_results,
        vif_results=vif_results,
    )

    return {
        "summary": summary,
        "vif_results": vif_results,
        "coef_results": coef_results,
    }


def run_kaggle_task() -> dict:
    kaggle_path = DATA_DIR / "kaggle_used_cars.csv"

    if not kaggle_path.exists():
        print("Error: Kaggle data file not found.")
        print("Please put your Kaggle CSV here:")
        print(kaggle_path)
        raise SystemExit(1)

    kaggle_df = clean_kaggle_data(kaggle_path)

    target_col = "price"
    numeric_cols = [
        "vehicle_age",
        "milage",
    ]
    categorical_cols = [
        "brand",
        "fuel_type",
        "transmission_simple",
        "accident",
        "clean_title",
    ]

    summary, fold_metrics = run_leakage_free_cv(
        df=kaggle_df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        positive_target=True,
    )

    coef_results, vif_results = fit_full_model(
        df=kaggle_df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    write_kaggle_report(
        summary=summary,
        fold_metrics=fold_metrics,
        coef_results=coef_results,
        vif_results=vif_results,
        original_rows=len(pd.read_csv(kaggle_path)),
        cleaned_rows=len(kaggle_df),
    )

    return {
        "summary": summary,
        "vif_results": vif_results,
        "coef_results": coef_results,
        "cleaned_rows": len(kaggle_df),
    }


def write_synthetic_report(
    summary: dict,
    fold_metrics: list[dict],
    coef_results: list[dict],
    vif_results: list[dict],
) -> None:
    report_path = RESULTS_DIR / "synthetic_report.md"

    content = "\n".join(
        [
            "# Week 11 Task A：模拟数据回归推测报告",
            "",
            "## 1. 数据生成机制 DGP",
            "",
            "本任务模拟一个广告投放与销售额之间的业务场景。每一行代表一个市场投放观测样本，目标变量是 `sales`。",
            "",
            "我设定的生成公式为：",
            "",
            "```text",
            "sales = 200 + 2.5 * tv_budget + 1.2 * online_budget + 0.15 * store_visits - 3.0 * competitor_price + region_effect + noise",
            "```",
            "",
            "其中：",
            "",
            "- `tv_budget`：电视广告预算，理论上正向影响销售额；",
            "- `online_budget`：线上广告预算，理论上正向影响销售额；",
            "- `store_visits`：门店访问量，理论上正向影响销售额；",
            "- `competitor_price`：竞争对手价格，理论上负向影响本企业销售额；",
            "- `region`：地区类别变量，不同地区有不同的销售基础水平。",
            "",
            "为了主动构造共线性，我先生成 `tv_budget`，再令：",
            "",
            "```text",
            "online_budget = 0.85 * tv_budget + 随机扰动",
            "```",
            "",
            "因此，`tv_budget` 和 `online_budget` 应该存在明显共线性。",
            "",
            "## 2. 人为加入的真实世界问题",
            "",
            "我在模拟数据中主动加入了以下问题：",
            "",
            "- 缺失值：部分 `tv_budget` 和 `competitor_price` 被设置为 NaN；",
            "- 异常值：部分 `online_budget` 和 `store_visits` 被放大；",
            "- 特征量纲差异：预算变量大约几十到几百，而 `store_visits` 大约上千；",
            "- 共线性：`tv_budget` 和 `online_budget` 高度相关。",
            "",
            "## 3. 无泄露 5 折交叉验证结果",
            "",
            make_metrics_table(summary),
            "",
            "### 每一折结果",
            "",
            make_fold_table(fold_metrics),
            "",
            "## 4. VIF 共线性诊断",
            "",
            make_vif_table(vif_results, n=10),
            "",
            "如果某些变量的 VIF 明显大于 10，说明存在严重多重共线性。在本模拟数据中，`tv_budget` 和 `online_budget` 是我主动构造的高度相关变量，因此它们的 VIF 较高是符合预期的。",
            "",
            "## 5. 系数方向推测",
            "",
            make_coef_table(coef_results, n=12),
            "",
            "从系数方向看，`tv_budget`、`online_budget` 和 `store_visits` 应该主要呈正向影响，`competitor_price` 应该主要呈负向影响。如果个别系数方向不稳定，主要原因通常是 `tv_budget` 和 `online_budget` 之间存在较强共线性，导致模型难以稳定地区分二者各自的独立贡献。",
            "",
            "## 6. 推测结论",
            "",
            "在模拟数据中，因为我知道真实的数据生成机制，所以可以检查模型推断是否与 DGP 一致。整体上，模型能够识别主要方向，但高度相关的广告预算变量会带来系数不稳定问题。这说明即使预测误差较低，也不能忽略共线性对解释性的影响。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def write_kaggle_report(
    summary: dict,
    fold_metrics: list[dict],
    coef_results: list[dict],
    vif_results: list[dict],
    original_rows: int,
    cleaned_rows: int,
) -> None:
    report_path = RESULTS_DIR / "kaggle_report.md"

    content = "\n".join(
        [
            "# Week 11 Task B：Kaggle 二手车价格预测报告",
            "",
            "## 1. 数据来源记录",
            "",
            "- 数据集名称：Used Car Price Prediction Dataset",
            "- Kaggle 页面链接：https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset",
            "- 使用文件：`kaggle_used_cars.csv`",
            "- 目标变量：`price`",
            "- 每一行样本代表：一辆二手车",
            f"- 原始样本量：{original_rows}",
            f"- 清洗后样本量：{cleaned_rows}",
            "",
            "## 2. 为什么选择这份数据？",
            "",
            "我选择这份二手车价格数据，是因为它是典型的回归问题。目标变量 `price` 是连续数值变量，业务问题是根据车辆年份、里程、品牌、燃料类型、变速箱、事故记录等信息预测二手车价格。",
            "",
            "这份数据不是过于简单的演示型数据。它包含字符串格式的价格和里程、缺失值、高基数类别变量、异常价格和真实业务字段，因此适合练习清洗、诊断、建模、无泄露交叉验证和业务解释。",
            "",
            "## 3. 数据清洗说明",
            "",
            "本次主要处理包括：",
            "",
            "- 将 `$10,300` 这种字符串价格转成数值型 `price`；",
            "- 将 `51,000 mi.` 这种字符串里程转成数值型 `milage`；",
            "- 根据 `model_year` 构造 `vehicle_age`；",
            "- 将复杂的变速箱字段简化为 `Automatic`、`Manual` 和 `Other`；",
            "- 将缺失的 `fuel_type`、`accident`、`clean_title` 填为 `Unknown` 类别；",
            "- 删除价格低于 1000 或高于 500000 的极端样本，避免极少数收藏级或异常车辆主导普通二手车价格模型；",
            "- 在交叉验证内部使用训练集学习类别合并、异常值截尾、均值填补和标准化参数。",
            "",
            "## 4. 无泄露 5 折交叉验证结果",
            "",
            make_metrics_table(summary),
            "",
            "### 每一折结果",
            "",
            make_fold_table(fold_metrics),
            "",
            "## 5. VIF 共线性诊断",
            "",
            make_vif_table(vif_results, n=10),
            "",
            "VIF 用于检查特征之间是否存在较强线性相关。如果某些变量 VIF 很高，说明这些变量之间可能存在信息重叠，系数解释需要谨慎。",
            "",
            "## 6. 主要系数方向",
            "",
            make_coef_table(coef_results, n=12),
            "",
            "## 7. 真实数据推测解释",
            "",
            f"根据无泄露交叉验证结果，模型预测二手车价格的平均绝对误差 MAE 约为 {summary['MAE']:.2f} 美元，平均百分比误差 MAPE 约为 {summary['MAPE']:.2f}%。",
            "",
            "这意味着，如果模型用于真实二手车估价，平均来看每辆车的预测价格会有一定金额误差。相比单纯追求更低分数，真实业务中更重要的是理解误差是否在可接受范围内，以及哪些车辆类型可能更难预测。",
            "",
            "## 8. 业务风险",
            "",
            "如果要上线，我最担心的风险包括：",
            "",
            "- 高端车或稀有车价格波动大，容易形成异常值主导；",
            "- `model`、`engine` 等高基数变量被简化或删除，可能损失细节信息；",
            "- 事故记录和 clean title 有缺失，可能影响价格解释；",
            "- 二手车市场价格随时间变化，模型需要定期更新。",
            "",
            "因此，这个模型可以作为初步估价参考，但不能完全替代人工评估或更详细的车辆检测。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def write_summary_report(synthetic_result: dict, kaggle_result: dict) -> None:
    report_path = RESULTS_DIR / "summary_comparison.md"

    synthetic_summary = synthetic_result["summary"]
    kaggle_summary = kaggle_result["summary"]

    content = "\n".join(
        [
            "# Week 11 Task C：模拟数据与 Kaggle 真实数据对照总结",
            "",
            "## 1. 指标对比",
            "",
            "| 数据类型 | RMSE | MAE | MAPE (%) |",
            "|---|---:|---:|---:|",
            f"| 模拟数据 | {synthetic_summary['RMSE']:.4f} | {synthetic_summary['MAE']:.4f} | {synthetic_summary['MAPE']:.4f} |",
            f"| Kaggle 二手车数据 | {kaggle_summary['RMSE']:.4f} | {kaggle_summary['MAE']:.4f} | {kaggle_summary['MAPE']:.4f} |",
            "",
            "## 2. 为什么模拟数据中的推测更容易？",
            "",
            "在模拟数据中，数据生成机制是我自己设定的，因此我知道哪些变量应该正向影响目标变量，哪些变量应该负向影响目标变量，也知道哪些变量存在共线性。这使得模型结果是否合理更容易判断。",
            "",
            "## 3. 为什么真实数据解释更困难？",
            "",
            "在 Kaggle 真实数据中，变量之间的关系不是由我控制的。二手车价格受到品牌、年份、里程、车况、事故记录、市场供需等多方面影响，而且很多字段存在缺失、异常和高基数类别。因此，即使模型分数看起来可以，系数解释也需要更加谨慎。",
            "",
            "## 4. 缺失值、异常值和共线性的差异",
            "",
            "模拟数据中的缺失值、异常值和共线性是人为设计的，因此它们的来源和影响比较清楚。真实数据中的问题则更复杂，例如极端价格可能代表豪车或录入错误，类别变量可能包含大量稀有车型，缺失值也可能本身带有业务含义。",
            "",
            "## 5. 为什么无泄露交叉验证在真实数据上尤其重要？",
            "",
            "真实数据中缺失值、异常值和类别分布更复杂。如果在交叉验证前先对全量数据做均值填补、标准化或类别合并，就会让验证集信息提前参与训练流程，导致评估结果偏乐观。无泄露交叉验证可以更接近模型上线后面对未知数据的情况。",
            "",
            "## 6. utils 工具箱带来的帮助",
            "",
            "本周我复用了自己的 `utils` 工具箱，包括：",
            "",
            "- `CustomOLS`：完成主要回归模型训练；",
            "- `calculate_rmse`、`calculate_mae`、`calculate_mape`：完成评估指标计算；",
            "- `CustomMeanImputer`、`CustomStandardScaler`、`Winsorizer`：完成缺失值填补、标准化和异常值处理；",
            "- `calculate_vif`：完成共线性诊断。",
            "",
            "这些工具减少了重复代码，也让我能够把模拟数据和真实数据放在同一套流程中比较。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_directories()

    print("===== Week 11 Assignment Started =====")

    print("\n[1/3] Running synthetic regression task...")
    synthetic_result = run_synthetic_task()
    print("Synthetic task finished.")

    print("\n[2/3] Running Kaggle used car regression task...")
    kaggle_result = run_kaggle_task()
    print("Kaggle task finished.")

    print("\n[3/3] Writing summary comparison report...")
    write_summary_report(synthetic_result, kaggle_result)
    print("Summary report finished.")

    print("\n===== Results saved to =====")
    print(RESULTS_DIR / "synthetic_report.md")
    print(RESULTS_DIR / "kaggle_report.md")
    print(RESULTS_DIR / "summary_comparison.md")
    print(DATA_DIR / "synthetic_regression.csv")


if __name__ == "__main__":
    main()