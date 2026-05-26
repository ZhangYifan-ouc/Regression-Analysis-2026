"""Week 10 Milestone Project 2: data leakage vs leakage-free CV.

Run from students/07_nc with:
    uv run src/milestone2/main.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make src/ importable when this script is executed directly.
STUDENT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = STUDENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import GradientDescentOLS
from utils.transformers import CustomStandardScaler

DATA_FILE_NAME = "dirty_q4_marketing.csv"
N_SPLITS = 5
RANDOM_SEED = 42

TARGET_CANDIDATES = (
    "Sales",
    "sales",
    "Revenue",
    "revenue",
    "Revenue_Generated",
    "revenue_generated",
    "Sales_Revenue",
    "sales_revenue",
    "ROI",
    "roi",
    "Profit",
    "profit",
    "Conversions",
    "conversions",
    "target",
    "Target",
    "y",
)

ID_OR_DATE_NAMES = {
    "id",
    "index",
    "row_id",
    "customer_id",
    "client_id",
    "campaign_id",
    "date",
    "day",
    "month",
    "timestamp",
}


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    rmse: float
    mae: float
    mape: float


@dataclass(frozen=True)
class CVResult:
    method_name: str
    fold_metrics: list[FoldMetrics]
    target_col: str
    data_path: Path
    n_rows: int
    n_features: int

    @property
    def average_metrics(self) -> dict[str, float]:
        return {
            "RMSE": float(np.mean([item.rmse for item in self.fold_metrics])),
            "MAE": float(np.mean([item.mae for item in self.fold_metrics])),
            "MAPE": float(np.nanmean([item.mape for item in self.fold_metrics])),
        }


@dataclass(frozen=True)
class PreprocessPlan:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_means: pd.Series
    categorical_modes: dict[str, object]
    dummy_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 10 leakage-free cross-validation homework.")
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV path. Default: search upward for data/dirty_q4_marketing.csv.",
    )
    return parser.parse_args()


def candidate_roots() -> Iterable[Path]:
    """Yield likely course-repository roots without hard-coded absolute paths."""
    anchors = [Path.cwd().resolve(), STUDENT_ROOT.resolve(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for anchor in anchors:
        for parent in [anchor, *anchor.parents]:
            if parent not in seen:
                seen.add(parent)
                yield parent


def resolve_data_path(optional_path: str | None = None) -> Path:
    """Find the teacher-provided data file from the course repository."""
    if optional_path:
        path = Path(optional_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"指定的数据文件不存在：{path}")

    env_path = os.environ.get("DIRTY_Q4_MARKETING_CSV")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"环境变量 DIRTY_Q4_MARKETING_CSV 指向的文件不存在：{path}")

    checked: list[Path] = []
    for root in candidate_roots():
        candidate = root / "data" / DATA_FILE_NAME
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n".join(f"- {path}" for path in checked[:16])
    raise FileNotFoundError(
        "找不到老师提供的 dirty_q4_marketing.csv。请先同步课程仓库，并保持 "
        "Regression-Analysis-2026/data/dirty_q4_marketing.csv 的相对位置；"
        "或使用 --data / DIRTY_Q4_MARKETING_CSV 指定数据路径。\n"
        f"已检查的部分候选路径：\n{checked_text}"
    )


def portable_data_label(data_path: Path) -> str:
    resolved = data_path.resolve()
    if resolved.name == DATA_FILE_NAME and resolved.parent.name == "data":
        return f"data/{DATA_FILE_NAME}"
    for root in candidate_roots():
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            pass
    return resolved.name


def reset_results_dir() -> Path:
    results_dir = STUDENT_ROOT / "week10" / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def read_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df


def choose_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            return col

    numeric_cols = [
        col
        for col in df.columns
        if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0
    ]
    if not numeric_cols:
        raise ValueError("数据中没有可用的数值型回归目标列。")
    return numeric_cols[-1]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    target_col = choose_target_column(df)
    y_series = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y_series.notna()
    if valid_mask.sum() < N_SPLITS:
        raise ValueError("有效目标值数量少于 5，无法进行 5 折交叉验证。")

    clean_df = df.loc[valid_mask].reset_index(drop=True)
    y = y_series.loc[valid_mask].to_numpy(dtype=float)
    X = clean_df.drop(columns=[target_col]).copy()

    removable_cols = [col for col in X.columns if col.strip().lower() in ID_OR_DATE_NAMES]
    X = X.drop(columns=removable_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("删除目标列和标识列后没有剩余特征。")
    return X, y, target_col


def make_folds(n_samples: int, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n_samples)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled_idx, n_splits)]


def fit_preprocess_plan(X_train: pd.DataFrame) -> PreprocessPlan:
    """Fit missing-value and one-hot parameters on a training frame only."""
    X_train = X_train.copy()
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X_train.columns:
        converted = pd.to_numeric(X_train[col], errors="coerce")
        non_missing = int(X_train[col].notna().sum())
        numeric_count = int(converted.notna().sum())
        if numeric_count > 0 and numeric_count / max(non_missing, 1) >= 0.5:
            X_train[col] = converted
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_means = pd.Series(dtype=float)
    if numeric_cols:
        numeric_means = X_train[numeric_cols].mean().fillna(0.0)

    categorical_modes: dict[str, object] = {}
    dummy_columns: list[str] = []
    if categorical_cols:
        categorical_frame = X_train[categorical_cols].copy()
        for col in categorical_cols:
            mode_values = categorical_frame[col].mode(dropna=True)
            categorical_modes[col] = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            categorical_frame[col] = categorical_frame[col].fillna(categorical_modes[col]).astype(str)
        dummies = pd.get_dummies(categorical_frame, columns=categorical_cols, drop_first=True, dtype=float)
        dummy_columns = dummies.columns.tolist()

    return PreprocessPlan(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_means=numeric_means,
        categorical_modes=categorical_modes,
        dummy_columns=dummy_columns,
    )


def transform_with_plan(X: pd.DataFrame, plan: PreprocessPlan) -> pd.DataFrame:
    """Apply a fitted preprocessing plan without learning from this frame."""
    parts: list[pd.DataFrame] = []

    if plan.numeric_cols:
        numeric_part = X.reindex(columns=plan.numeric_cols).copy()
        for col in plan.numeric_cols:
            numeric_part[col] = pd.to_numeric(numeric_part[col], errors="coerce")
        numeric_part = numeric_part.fillna(plan.numeric_means).astype(float)
        parts.append(numeric_part)

    if plan.categorical_cols:
        categorical_part = X.reindex(columns=plan.categorical_cols).copy()
        for col in plan.categorical_cols:
            categorical_part[col] = categorical_part[col].fillna(plan.categorical_modes[col]).astype(str)
        dummies = pd.get_dummies(categorical_part, columns=plan.categorical_cols, drop_first=True, dtype=float)
        dummies = dummies.reindex(columns=plan.dummy_columns, fill_value=0.0)
        parts.append(dummies)

    if not parts:
        raise ValueError("预处理后没有可用特征。")

    prepared = pd.concat(parts, axis=1)
    prepared = prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prepared.astype(float)


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def train_gradient_descent_model(X_train: np.ndarray, y_train: np.ndarray) -> GradientDescentOLS:
    model = GradientDescentOLS(
        learning_rate=0.01,
        tol=1e-8,
        max_iter=12000,
        gd_type="full_batch",
    )
    return model.fit(add_intercept(X_train), y_train, seed=RANDOM_SEED)


def collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def bad_cross_validation(data_path: Path) -> CVResult:
    """Intentionally leaky CV: global preprocessing happens before 5-fold CV."""
    df = read_data(data_path)
    X_raw, y, target_col = split_features_target(df)

    # Data leakage: the imputer, dummy schema, and scaler all see the full X.
    global_plan = fit_preprocess_plan(X_raw)
    X_prepared = transform_with_plan(X_raw, global_plan)
    global_scaler = CustomStandardScaler()
    X_scaled = global_scaler.fit_transform(X_prepared.to_numpy(dtype=float))

    folds = make_folds(len(y))
    fold_metrics: list[FoldMetrics] = []
    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = train_gradient_descent_model(X_train, y_train)
        y_pred = model.predict(add_intercept(X_val))
        metrics = collect_metrics(y_val, y_pred)
        fold_metrics.append(FoldMetrics(fold_id, metrics["RMSE"], metrics["MAE"], metrics["MAPE"]))

    result = CVResult(
        method_name="Bad CV：全局预处理后交叉验证（有数据泄露）",
        fold_metrics=fold_metrics,
        target_col=target_col,
        data_path=data_path,
        n_rows=len(y),
        n_features=X_scaled.shape[1],
    )
    print(f"[Bad CV] mean RMSE = {result.average_metrics['RMSE']:.4f}")
    return result


def good_cross_validation(data_path: Path) -> CVResult:
    """Leakage-free CV: preprocessing is fitted inside each training fold only."""
    df = read_data(data_path)
    X_raw, y, target_col = split_features_target(df)
    folds = make_folds(len(y))

    fold_metrics: list[FoldMetrics] = []
    max_n_features = 0
    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Fit missing-value and dummy parameters on X_train only.
        fold_plan = fit_preprocess_plan(X_train_raw)
        X_train_prepared = transform_with_plan(X_train_raw, fold_plan)
        X_val_prepared = transform_with_plan(X_val_raw, fold_plan)

        # Fit scaler on X_train only; validation set only calls transform().
        fold_scaler = CustomStandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train_prepared.to_numpy(dtype=float))
        X_val_scaled = fold_scaler.transform(X_val_prepared.to_numpy(dtype=float))
        max_n_features = max(max_n_features, X_train_scaled.shape[1])

        model = train_gradient_descent_model(X_train_scaled, y_train)
        y_pred = model.predict(add_intercept(X_val_scaled))
        metrics = collect_metrics(y_val, y_pred)
        fold_metrics.append(FoldMetrics(fold_id, metrics["RMSE"], metrics["MAE"], metrics["MAPE"]))

    result = CVResult(
        method_name="Good CV：折内预处理流水线（无数据泄露）",
        fold_metrics=fold_metrics,
        target_col=target_col,
        data_path=data_path,
        n_rows=len(y),
        n_features=max_n_features,
    )
    print(f"[Good CV] mean RMSE = {result.average_metrics['RMSE']:.4f}")
    return result


def fmt(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    return f"{value:,.4f}"


def comparison_table(bad: CVResult, good: CVResult) -> str:
    bad_avg = bad.average_metrics
    good_avg = good.average_metrics
    return "\n".join(
        [
            "| 方法 | RMSE | MAE | MAPE(%) |",
            "|---|---:|---:|---:|",
            f"| Bad CV（有数据泄露） | {fmt(bad_avg['RMSE'])} | {fmt(bad_avg['MAE'])} | {fmt(bad_avg['MAPE'])} |",
            f"| Good CV（无数据泄露） | {fmt(good_avg['RMSE'])} | {fmt(good_avg['MAE'])} | {fmt(good_avg['MAPE'])} |",
        ]
    )


def fold_table(bad: CVResult, good: CVResult) -> str:
    rows = [
        "| Fold | Bad RMSE | Bad MAE | Bad MAPE(%) | Good RMSE | Good MAE | Good MAPE(%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bad_fold, good_fold in zip(bad.fold_metrics, good.fold_metrics, strict=True):
        rows.append(
            f"| {bad_fold.fold} | {fmt(bad_fold.rmse)} | {fmt(bad_fold.mae)} | {fmt(bad_fold.mape)} | "
            f"{fmt(good_fold.rmse)} | {fmt(good_fold.mae)} | {fmt(good_fold.mape)} |"
        )
    return "\n".join(rows)


def write_evaluation_comparison(results_dir: Path, bad: CVResult, good: CVResult) -> None:
    bad_avg = bad.average_metrics
    good_avg = good.average_metrics
    rmse_gap = good_avg["RMSE"] - bad_avg["RMSE"]
    mae_gap = good_avg["MAE"] - bad_avg["MAE"]
    mape_gap = good_avg["MAPE"] - bad_avg["MAPE"]

    report = f"""# Week 10 Evaluation Comparison

## 1. 数据与设置

- 数据来源：`{portable_data_label(bad.data_path)}`
- 回归目标列：`{bad.target_col}`
- 有效样本数：{bad.n_rows}
- 交叉验证：5-Fold CV，`shuffle=True` 的等价随机切分，随机种子 {RANDOM_SEED}
- Bad CV 特征数：{bad.n_features}
- Good CV 最大折内特征数：{good.n_features}

## 2. 平均指标对比

{comparison_table(bad, good)}

## 3. 分折结果

{fold_table(bad, good)}

## 4. 思考题：为什么 Bad CV 的“好看”是致命的？

Bad CV 在切分交叉验证之前，已经用全量数据学习了缺失值填补均值、类别编码结构和标准化均值/标准差。也就是说，验证折的分布信息提前进入了训练流程。这样的 RMSE、MAE、MAPE 可能会显得更低，但它评估的不是模型面对未知样本时的真实表现。

Good CV 把所有会学习参数的步骤放进每一折的训练循环内部：先切分 `X_train` 和 `X_val`，再只用 `X_train` 学习填补参数和 scaler 参数；验证集只调用 `transform()`。因此 Good CV 的分数更接近模型上线后的真实泛化误差。

本次运行中，Good CV 相比 Bad CV 的平均 RMSE 差值为 **{fmt(rmse_gap)}**，MAE 差值为 **{fmt(mae_gap)}**，MAPE 差值为 **{fmt(mape_gap)}** 个百分点。无论差距大小，最终汇报都应以 Good CV 为准，因为它遵守了验证集隔离原则。

## 5. CMO 视角：业务解释

如果 `{bad.target_col}` 表示广告预算、营销收益或销售额，那么 Good CV 的 MAE 可以理解为上线后单条预测平均会偏离真实值约 **{fmt(good_avg['MAE'])}** 个金额单位；Good CV 的 MAPE 约为 **{fmt(good_avg['MAPE'])}%**，表示平均相对误差约为这个百分比。

因此，应该给老板看 Good CV 的“更保守成绩”，而不是 Bad CV 的“漂亮成绩”。Bad CV 会低估风险，可能让业务团队过度相信模型，进而错误扩大投放或低估预算误差。
"""
    (results_dir / "evaluation_comparison.md").write_text(report, encoding="utf-8")


def write_detailed_answer(results_dir: Path, bad: CVResult, good: CVResult) -> None:
    report = f"""# Week 10 作业详细解答

## Task 1：评估指标库 `utils/metrics.py`

我手写实现了三个回归预测指标：

1. `calculate_rmse(y_true, y_pred)`：计算均方误差后开平方，对大误差更加敏感；
2. `calculate_mae(y_true, y_pred)`：计算绝对误差平均值，单位和目标变量一致，便于向业务解释；
3. `calculate_mape(y_true, y_pred)`：计算平均绝对百分比误差。代码使用 `epsilon=1e-8` 过滤真实值为 0 或极小的样本，避免除零导致无穷大或误导性百分比。

## Task 2：转换器 API `utils/transformers.py`

`CustomStandardScaler` 按 Transformer 规范实现：

- `fit(X)`：只计算并保存 `self.mean_` 和 `self.std_`；
- `transform(X)`：只复用已经保存的均值和标准差，不重新学习；
- `fit_transform(X)`：组合调用 `fit()` 和 `transform()`。

这个接口的关键是把“学习参数”和“应用参数”分开。Good CV 中验证集只能使用训练集学到的参数，所以验证集不会参与 scaler 的 `fit()`。

## Task 3：Bad Cross Validation（故意制造数据泄露）

`bad_cross_validation()` 的流程是：

1. 读取老师放在 `data/dirty_q4_marketing.csv` 的脏数据；
2. 在全量 `X` 上计算缺失值填补参数、类别编码结构和标准化参数；
3. 对全量数据先完成 `fit_transform()`；
4. 再把处理好的数据送入 5 折交叉验证。

这一步故意制造数据泄露，因为验证折的信息提前参与了全局预处理参数的学习。

本次 Bad CV 平均指标：

- RMSE：{fmt(bad.average_metrics['RMSE'])}
- MAE：{fmt(bad.average_metrics['MAE'])}
- MAPE：{fmt(bad.average_metrics['MAPE'])}%

## Task 4：Good Cross Validation（无泄露流水线）

`good_cross_validation()` 的每一折都重新执行一条独立流水线：

1. 先切分 `X_train` 和 `X_val`；
2. 只在 `X_train` 上学习缺失值填补均值和类别编码结构；
3. 只在 `X_train` 上调用 `CustomStandardScaler.fit_transform()`；
4. 对 `X_val` 只调用已经拟合好的 `transform()`；
5. 用 `GradientDescentOLS` 在训练折训练，并在验证折上计算 RMSE、MAE、MAPE。

本次 Good CV 平均指标：

- RMSE：{fmt(good.average_metrics['RMSE'])}
- MAE：{fmt(good.average_metrics['MAE'])}
- MAPE：{fmt(good.average_metrics['MAPE'])}%

## Task 5：自动化 I/O 与制品管理

程序唯一入口为：

```bash
uv run src/milestone2/main.py
```

程序启动时会自动清空并重建 `week10/results/` 文件夹，输出：

- `week10/results/evaluation_comparison.md`：Bad CV 和 Good CV 的指标对比；
- `week10/results/week10.md`：本文件，详细作业解答；
- `week10/results/leakage_analysis.png`：有无泄露的误差柱状图。

## Task 6：课堂答辩提纲

### CTO 视角

展示 `CustomStandardScaler` 的 `fit/transform/fit_transform` 结构。重点说明 Good CV 每一折都会重新创建 `fold_plan` 和 `fold_scaler`，并且只用训练折调用 `fit()`；验证折只调用 `transform()`，没有任何二次拟合。

### CMO 视角

更应该汇报 Good CV。按本次运行结果，模型上线后目标列 `{bad.target_col}` 的平均绝对误差约为 **{fmt(good.average_metrics['MAE'])}**，平均相对误差约为 **{fmt(good.average_metrics['MAPE'])}%**。这比 Bad CV 更保守，但更接近真实上线环境，因此更适合做预算预测和投放决策。
"""
    (results_dir / "week10.md").write_text(report, encoding="utf-8")


def plot_leakage_analysis(results_dir: Path, bad: CVResult, good: CVResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib 不可用，跳过绘图：{exc}")
        return

    labels = ["RMSE", "MAE", "MAPE"]
    bad_values = [bad.average_metrics[label] for label in labels]
    good_values = [good.average_metrics[label] for label in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bad_values, width, label="Bad CV (leakage)")
    ax.bar(x + width / 2, good_values, width, label="Good CV (leakage-free)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error")
    ax.set_title("Week 10: Leakage vs Leakage-Free CV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "leakage_analysis.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = reset_results_dir()
    data_path = resolve_data_path(args.data)
    print(f"读取数据：{portable_data_label(data_path)}")

    bad_result = bad_cross_validation(data_path)
    good_result = good_cross_validation(data_path)

    write_evaluation_comparison(results_dir, bad_result, good_result)
    write_detailed_answer(results_dir, bad_result, good_result)
    plot_leakage_analysis(results_dir, bad_result, good_result)

    print(f"结果已生成：{results_dir}")
    print("- evaluation_comparison.md")
    print("- week10.md")
    print("- leakage_analysis.png")


if __name__ == "__main__":
    main()
