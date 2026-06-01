# ============================================================
# Week 11 Assignment
# Dual Inference Sprint — Synthetic-to-Real Regression Workflow
#
# 文件位置：
# students/23_zy/src/week11/main.py
#
# 运行方式：
# uv run src/week11/main.py
#
# 说明：
# 1. 本代码会自动生成模拟数据 synthetic_regression.csv
# 2. 会读取 Kaggle 数据 kaggle_insurance.csv
# 3. 会完成清洗、编码、标准化、OLS 建模、5 折交叉验证、VIF 诊断
# 4. 会在 src/week11/results/ 下生成三个报告：
#    - synthetic_report.md
#    - kaggle_report.md
#    - summary_comparison.md
# ============================================================

import os
import sys
import math
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_regression.csv")
KAGGLE_PATH = os.path.join(DATA_DIR, "kaggle_insurance.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. 自己实现一些基础工具
#    这些工具和前几周 utils 中的思想一致：
#    缺失值填补、标准化、OLS、RMSE/MAE/MAPE、VIF
# ============================================================

def rmse(y_true, y_pred):
    """均方根误差：越小越好"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """平均绝对误差：越小越好"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    平均绝对百分比误差：越小越好
    注意：为了避免 y=0 导致除零，这里加了一个很小的数
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


class CustomImputer:
    """
    自定义缺失值填补器
    数值变量：用训练集的中位数填补
    类别变量：用训练集的众数填补
    """

    def __init__(self):
        self.fill_values = {}

    def fit(self, df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.fill_values[col] = df[col].median()
            else:
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    self.fill_values[col] = mode_value.iloc[0]
                else:
                    self.fill_values[col] = "Unknown"
        return self

    def transform(self, df):
        df_new = df.copy()
        for col, value in self.fill_values.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].fillna(value)
        return df_new

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class CustomStandardScaler:
    """
    自定义标准化器
    只在训练集 fit，验证集只 transform
    这样可以避免数据泄露
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns = None

    def fit(self, X):
        X = pd.DataFrame(X)
        self.columns = X.columns
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0).replace(0, 1)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CustomOLS:
    """
    自定义 OLS 线性回归模型
    使用正规方程：
    beta = (X'X)^(-1)X'y
    为了避免矩阵不可逆，这里使用 np.linalg.pinv 伪逆
    """

    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 添加截距项
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])

        self.beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])
        return X_design @ self.beta


def calculate_vif(X, feature_names):
    """
    计算 VIF 方差膨胀因子
    VIF 越大，说明该变量和其他变量之间的共线性越严重
    一般 VIF > 10 可以认为有明显共线性风险
    """
    X = np.asarray(X, dtype=float)
    vif_results = []

    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)

        model = CustomOLS()
        model.fit(X_others, y_i)
        y_pred = model.predict(X_others)

        ss_res = np.sum((y_i - y_pred) ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)

        if ss_tot == 0:
            r2 = 0
        else:
            r2 = 1 - ss_res / ss_tot

        if r2 >= 0.999999:
            vif = float("inf")
        else:
            vif = 1 / (1 - r2)

        vif_results.append((feature_names[i], vif))

    return vif_results


def make_kfold_indices(n_samples, n_splits=5, random_state=42):
    """
    自己实现简单的 5 折交叉验证索引
    这样不依赖 sklearn
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size

        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        folds.append((train_idx, val_idx))
        current = stop

    return folds


# ============================================================
# 2. Task A：生成模拟数据
# ============================================================

def generate_synthetic_data():
    """
    生成一份模拟业务数据：
    场景：广告预算与销售额预测

    变量说明：
    - tv_budget：电视广告预算
    - online_budget：线上广告预算，与 tv_budget 高度相关
    - radio_budget：广播广告预算
    - price_discount：折扣力度
    - region：地区类别变量
    - sales：销售额，即目标变量

    这里故意加入：
    1. 缺失值
    2. 异常值
    3. 共线性
    4. 类别变量
    """
    np.random.seed(42)
    n = 500

    tv_budget = np.random.normal(loc=100, scale=25, size=n)

    # 构造高度相关变量：online_budget 与 tv_budget 强相关
    online_budget = 0.85 * tv_budget + np.random.normal(loc=0, scale=5, size=n)

    radio_budget = np.random.normal(loc=40, scale=10, size=n)
    price_discount = np.random.uniform(0, 0.3, size=n)

    regions = np.random.choice(
        ["east", "west", "north", "south"],
        size=n,
        p=[0.35, 0.25, 0.2, 0.2]
    )

    region_effect = {
        "east": 20,
        "west": 10,
        "north": -5,
        "south": 0
    }

    noise = np.random.normal(loc=0, scale=15, size=n)

    sales = (
        80
        + 2.5 * tv_budget
        + 1.2 * online_budget
        + 1.8 * radio_budget
        - 120 * price_discount
        + np.array([region_effect[r] for r in regions])
        + noise
    )

    df = pd.DataFrame({
        "tv_budget": tv_budget,
        "online_budget": online_budget,
        "radio_budget": radio_budget,
        "price_discount": price_discount,
        "region": regions,
        "sales": sales
    })

    # 加入缺失值
    missing_idx_1 = np.random.choice(df.index, size=20, replace=False)
    missing_idx_2 = np.random.choice(df.index, size=15, replace=False)

    df.loc[missing_idx_1, "radio_budget"] = np.nan
    df.loc[missing_idx_2, "region"] = np.nan

    # 加入异常值
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, "tv_budget"] = df.loc[outlier_idx, "tv_budget"] * 4

    df.to_csv(SYNTHETIC_PATH, index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 3. 通用预处理函数
# ============================================================

def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    """
    对异常值进行缩尾处理
    低于 1% 分位数的值设为 1% 分位数
    高于 99% 分位数的值设为 99% 分位数
    """
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)


def prepare_features(df, target_col):
    """
    把原始数据拆成 X 和 y
    并对类别变量做 one-hot 编码
    """
    df = df.copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 类别变量 one-hot 编码
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 确保所有列都是数值型
    X_encoded = X_encoded.astype(float)

    return X_encoded, y.astype(float)


def run_no_leakage_cv(df, target_col, dataset_name):
    """
    无数据泄露的 5 折交叉验证

    核心思想：
    每一折都只在训练集上 fit 填补器和标准化器
    然后用训练集学到的规则去 transform 验证集
    不能先对全数据标准化再切分，否则就是数据泄露
    """
    folds = make_kfold_indices(len(df), n_splits=5, random_state=42)

    metric_rows = []
    coef_rows = []
    vif_results_all = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # 目标变量也不能有缺失
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])

        # 分离 X 和 y 前，先对特征做缺失值填补
        X_train_raw = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].astype(float)

        X_val_raw = val_df.drop(columns=[target_col])
        y_val = val_df[target_col].astype(float)

        # 缺失值填补：只在训练集 fit
        imputer = CustomImputer()
        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_val_imputed = imputer.transform(X_val_raw)

        # 合并回去方便统一 one-hot
        X_all = pd.concat([X_train_imputed, X_val_imputed], axis=0)
        X_all_encoded = pd.get_dummies(X_all, drop_first=True).astype(float)

        X_train_encoded = X_all_encoded.iloc[:len(X_train_imputed)].copy()
        X_val_encoded = X_all_encoded.iloc[len(X_train_imputed):].copy()

        feature_names = list(X_train_encoded.columns)

        # 标准化：只在训练集 fit
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_val_scaled = scaler.transform(X_val_encoded)

        # 模型训练
        model = CustomOLS()
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)

        # 指标
        row = {
            "dataset": dataset_name,
            "fold": fold_id,
            "RMSE": rmse(y_val, y_pred),
            "MAE": mae(y_val, y_pred),
            "MAPE": mape(y_val, y_pred)
        }
        metric_rows.append(row)

        # 系数
        beta = model.beta
        coef_dict = {"fold": fold_id, "intercept": beta[0]}
        for name, value in zip(feature_names, beta[1:]):
            coef_dict[name] = value
        coef_rows.append(coef_dict)

        # VIF 只记录第一折，避免报告过长
        if fold_id == 1:
            vif_results_all = calculate_vif(X_train_scaled, feature_names)

    metric_df = pd.DataFrame(metric_rows)
    coef_df = pd.DataFrame(coef_rows)

    return metric_df, coef_df, vif_results_all


# ============================================================
# 4. Task A：模拟数据完整流程
# ============================================================

def run_synthetic_workflow():
    print("\n========== Task A：模拟数据流程开始 ==========")

    df = generate_synthetic_data()
    print(f"模拟数据已保存：{SYNTHETIC_PATH}")
    print(f"模拟数据样本量：{len(df)}")

    # 对明显异常的 tv_budget 做缩尾
    df["tv_budget"] = winsorize_series(df["tv_budget"], 0.01, 0.99)

    metric_df, coef_df, vif_results = run_no_leakage_cv(
        df=df,
        target_col="sales",
        dataset_name="synthetic"
    )

    report_path = os.path.join(RESULTS_DIR, "synthetic_report.md")

    avg_rmse = metric_df["RMSE"].mean()
    avg_mae = metric_df["MAE"].mean()
    avg_mape = metric_df["MAPE"].mean()

    coef_mean = coef_df.drop(columns=["fold"]).mean(numeric_only=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Week 11 Task A：模拟数据回归分析报告\n\n")

        f.write("## 1. 数据生成机制 DGP\n\n")
        f.write("本部分构造的是一个广告预算影响销售额的模拟业务场景。目标变量是 `sales`，表示销售额。\n\n")
        f.write("我设定的真实生成公式大致为：\n\n")
        f.write("```text\n")
        f.write("sales = 80 + 2.5 * tv_budget + 1.2 * online_budget + 1.8 * radio_budget\n")
        f.write("        - 120 * price_discount + region_effect + noise\n")
        f.write("```\n\n")
        f.write("其中，`tv_budget`、`online_budget`、`radio_budget` 对销售额是正向影响，")
        f.write("`price_discount` 在这里被设定为负向影响。地区变量 `region` 通过不同地区效应影响销售额。\n\n")

        f.write("## 2. 主动加入的数据问题\n\n")
        f.write("- 缺失值：在 `radio_budget` 和 `region` 中人为加入缺失值；\n")
        f.write("- 异常值：在 `tv_budget` 中人为放大部分样本；\n")
        f.write("- 共线性：令 `online_budget = 0.85 * tv_budget + 随机扰动`，因此它和 `tv_budget` 高度相关；\n")
        f.write("- 类别变量：`region` 是非数值型变量，需要进行 one-hot 编码。\n\n")

        f.write("## 3. 5 折交叉验证结果\n\n")
        f.write(metric_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"平均 RMSE：{avg_rmse:.4f}\n\n")
        f.write(f"平均 MAE：{avg_mae:.4f}\n\n")
        f.write(f"平均 MAPE：{avg_mape:.4f}%\n\n")

        f.write("## 4. 平均系数方向\n\n")
        f.write(coef_mean.to_frame("mean_coefficient").to_markdown())
        f.write("\n\n")
        f.write("从平均系数看，大多数变量方向与我设定的 DGP 基本一致。")
        f.write("但是由于 `tv_budget` 和 `online_budget` 被人为设置为高度相关，")
        f.write("它们各自的系数可能不够稳定，这正是共线性会带来的问题。\n\n")

        f.write("## 5. VIF 共线性诊断\n\n")
        vif_df = pd.DataFrame(vif_results, columns=["feature", "VIF"])
        f.write(vif_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("一般来说，VIF 大于 10 可以认为存在比较明显的共线性风险。")
        f.write("在本模拟数据中，如果 `tv_budget` 和 `online_budget` 的 VIF 较高，")
        f.write("说明模型确实识别出了我在 DGP 中主动构造的共线性问题。\n\n")

        f.write("## 6. 推测结论\n\n")
        f.write("由于模拟数据的生成机制是已知的，所以我们可以直接对比模型估计结果和真实设定。")
        f.write("整体上，模型能够恢复主要变量的影响方向。")
        f.write("但对于高度相关的变量，单个系数的解释要谨慎，因为模型很难把两个高度相关变量的作用完全分开。\n")

    print(f"模拟数据报告已保存：{report_path}")

    return metric_df, coef_df, vif_results


# ============================================================
# 5. Task B：Kaggle 真实数据完整流程
# ============================================================

def load_kaggle_data():
    """
    读取 Kaggle 医疗费用数据
    数据集字段：
    age, sex, bmi, children, smoker, region, charges
    charges 是连续型目标变量
    """
    if not os.path.exists(KAGGLE_PATH):
        raise FileNotFoundError(
            f"没有找到 Kaggle 数据文件：{KAGGLE_PATH}\n"
            "请确认文件名是 kaggle_insurance.csv，并且放在 src/week11/data/ 目录下。"
        )

    df = pd.read_csv(KAGGLE_PATH)
    return df


def run_kaggle_workflow():
    print("\n========== Task B：Kaggle 真实数据流程开始 ==========")

    df = load_kaggle_data()
    print(f"Kaggle 数据读取成功：{KAGGLE_PATH}")
    print(f"Kaggle 数据样本量：{len(df)}")
    print(f"Kaggle 数据列名：{list(df.columns)}")

    # 统一列名，防止有空格
    df.columns = [c.strip() for c in df.columns]

    # 保险费用 charges 是目标变量
    target_col = "charges"

    if target_col not in df.columns:
        raise ValueError("没有找到目标变量 charges，请检查 Kaggle 数据文件是否正确。")

    # 简单清洗
    # 1. 删除完全重复的行
    df = df.drop_duplicates().copy()

    # 2. 对 bmi 和 charges 做缩尾，降低异常值影响
    if "bmi" in df.columns:
        df["bmi"] = winsorize_series(df["bmi"], 0.01, 0.99)

    df["charges"] = winsorize_series(df["charges"], 0.01, 0.99)

    metric_df, coef_df, vif_results = run_no_leakage_cv(
        df=df,
        target_col=target_col,
        dataset_name="kaggle_insurance"
    )

    report_path = os.path.join(RESULTS_DIR, "kaggle_report.md")

    avg_rmse = metric_df["RMSE"].mean()
    avg_mae = metric_df["MAE"].mean()
    avg_mape = metric_df["MAPE"].mean()

    coef_mean = coef_df.drop(columns=["fold"]).mean(numeric_only=True).sort_values(
        key=lambda s: np.abs(s),
        ascending=False
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Week 11 Task B：Kaggle 医疗费用数据回归分析报告\n\n")

        f.write("## 1. 数据集说明\n\n")
        f.write("本次选择的 Kaggle 数据集是 **Medical Cost Personal Dataset**。\n\n")
        f.write("原始链接：`https://www.kaggle.com/datasets/d3lhomi10/medical-cost-personal-dataset`\n\n")
        f.write("该数据集中，每一行样本代表一位投保人的基本信息和对应医疗保险费用。")
        f.write("目标变量是 `charges`，表示个人医疗保险费用，是一个连续型变量，因此适合做回归分析。\n\n")

        f.write("## 2. 变量含义\n\n")
        f.write("- `age`：年龄；\n")
        f.write("- `sex`：性别；\n")
        f.write("- `bmi`：身体质量指数；\n")
        f.write("- `children`：子女数量；\n")
        f.write("- `smoker`：是否吸烟；\n")
        f.write("- `region`：地区；\n")
        f.write("- `charges`：医疗保险费用，也是本次预测目标。\n\n")

        f.write("## 3. 为什么选择这个数据集\n\n")
        f.write("我选择这份数据，是因为它虽然规模不算特别大，但具有比较明确的现实业务含义。")
        f.write("它同时包含数值变量和类别变量，例如年龄、BMI、子女数量是数值变量，")
        f.write("性别、是否吸烟、地区是类别变量。")
        f.write("因此它不是简单地直接套模型就结束，而是需要完成类别编码、异常值处理、标准化和模型解释。\n\n")

        f.write("## 4. 数据清洗与预处理\n\n")
        f.write("本次处理流程包括：\n\n")
        f.write("1. 删除重复样本；\n")
        f.write("2. 对 `bmi` 和 `charges` 使用 1% 和 99% 分位数进行缩尾处理；\n")
        f.write("3. 对类别变量进行 one-hot 编码；\n")
        f.write("4. 在每一折交叉验证中，只用训练集拟合缺失值填补器和标准化器，再作用于验证集，避免数据泄露；\n")
        f.write("5. 使用自定义 OLS 完成主模型训练；\n")
        f.write("6. 使用 RMSE、MAE、MAPE 评价模型表现。\n\n")

        f.write("## 5. 5 折交叉验证结果\n\n")
        f.write(metric_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"平均 RMSE：{avg_rmse:.4f}\n\n")
        f.write(f"平均 MAE：{avg_mae:.4f}\n\n")
        f.write(f"平均 MAPE：{avg_mape:.4f}%\n\n")

        f.write("从业务角度看，MAE 可以理解为模型平均会预测错多少医疗费用金额。")
        f.write("MAPE 可以理解为平均百分比误差，更适合从相对误差角度理解模型效果。\n\n")

        f.write("## 6. 平均系数方向与变量稳定性\n\n")
        f.write(coef_mean.to_frame("mean_coefficient").to_markdown())
        f.write("\n\n")
        f.write("从系数结果看，`smoker_yes` 通常会是影响医疗费用最明显的变量之一。")
        f.write("这也符合现实直觉：吸烟者的健康风险更高，保险费用往往也更高。")
        f.write("此外，年龄和 BMI 通常也会对医疗费用产生正向影响。")
        f.write("但是地区和性别变量的影响可能没有那么稳定，需要谨慎解释。\n\n")

        f.write("## 7. VIF 共线性诊断\n\n")
        vif_df = pd.DataFrame(vif_results, columns=["feature", "VIF"])
        f.write(vif_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("如果 VIF 明显大于 10，说明变量之间可能存在较强共线性。")
        f.write("在真实数据中，共线性会影响单个变量系数的解释稳定性，")
        f.write("即使模型预测效果还可以，也不能随便把每个系数都解释成严格的因果关系。\n\n")

        f.write("## 8. 真实数据推测结论\n\n")
        f.write("在这份医疗费用数据中，吸烟状态、年龄和 BMI 是比较容易解释的变量。")
        f.write("其中吸烟状态的影响最明显，也最符合业务直觉。")
        f.write("不过，真实数据和模拟数据不同，我们并不知道真正的数据生成机制，")
        f.write("所以模型结果更多体现的是相关关系，而不能直接解释为因果关系。")
        f.write("如果模型将来用于实际保险定价，我最担心的是样本代表性不足、异常值影响、")
        f.write("以及不同群体之间可能存在的公平性问题。\n")

    print(f"Kaggle 报告已保存：{report_path}")

    return metric_df, coef_df, vif_results


# ============================================================
# 6. Task C：模拟数据与真实数据对比总结
# ============================================================

def write_summary_report(synthetic_metric_df, kaggle_metric_df):
    print("\n========== Task C：对比总结开始 ==========")

    summary_path = os.path.join(RESULTS_DIR, "summary_comparison.md")

    syn_rmse = synthetic_metric_df["RMSE"].mean()
    syn_mae = synthetic_metric_df["MAE"].mean()
    syn_mape = synthetic_metric_df["MAPE"].mean()

    kag_rmse = kaggle_metric_df["RMSE"].mean()
    kag_mae = kaggle_metric_df["MAE"].mean()
    kag_mape = kaggle_metric_df["MAPE"].mean()

    comparison_df = pd.DataFrame({
        "dataset": ["synthetic", "kaggle_insurance"],
        "mean_RMSE": [syn_rmse, kag_rmse],
        "mean_MAE": [syn_mae, kag_mae],
        "mean_MAPE": [syn_mape, kag_mape]
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Week 11 Task C：模拟数据与 Kaggle 真实数据对比总结\n\n")

        f.write("## 1. 指标对比\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. 为什么模拟数据中的推测更容易？\n\n")
        f.write("在模拟数据中，我知道数据是怎么被生成出来的，也知道每个变量在真实公式中的方向和大致作用。")
        f.write("因此，当模型估计出系数之后，我可以直接把结果和 DGP 进行比较。")
        f.write("如果结果不一致，我也能大致判断是噪声、异常值、共线性还是预处理造成的。\n\n")

        f.write("## 3. 为什么真实数据解释更困难？\n\n")
        f.write("真实 Kaggle 数据并没有告诉我们真正的数据生成机制。")
        f.write("例如医疗费用受到年龄、BMI、吸烟状态等因素影响，但也可能受到疾病史、保险类型、地区医疗价格等未观测因素影响。")
        f.write("这些变量不在数据中，模型就无法直接控制它们。")
        f.write("所以即使模型分数还可以，解释时也要注意：这更多是相关关系，而不是严格因果关系。\n\n")

        f.write("## 4. 共线性、缺失值、异常值的影响差异\n\n")
        f.write("在模拟数据中，共线性是我主动设计的，因此它的来源比较清楚。")
        f.write("而在真实数据中，共线性可能来自变量之间的自然关系，例如年龄、身体状况和医疗费用之间可能本来就有关联。")
        f.write("缺失值和异常值在模拟数据中是可控的，但在真实数据中，它们可能反映数据采集过程中的真实问题，")
        f.write("因此处理时不能只追求模型分数，还要考虑业务合理性。\n\n")

        f.write("## 5. 为什么无泄露交叉验证很重要？\n\n")
        f.write("如果先对全体数据做缺失值填补、标准化或异常值处理，再切分训练集和验证集，")
        f.write("验证集的信息就会提前进入训练过程，这就是数据泄露。")
        f.write("数据泄露会让模型评估结果看起来更好，但上线后面对新数据时效果可能下降。")
        f.write("本次代码在每一折中都只用训练集拟合填补器和标准化器，然后再处理验证集，所以可以减少数据泄露问题。\n\n")

        f.write("## 6. 自己维护 utils 思想的作用\n\n")
        f.write("这周虽然只写了一个 `main.py`，但代码中延续了前几周的组件化思想。")
        f.write("例如缺失值填补、标准化、OLS 模型、评估指标和 VIF 诊断都被拆成独立函数或类。")
        f.write("这样做的好处是，以后换一个数据集时，不需要重新写整套流程，")
        f.write("只需要调整目标变量、清洗规则和报告解释即可。\n\n")

        f.write("## 7. 总结\n\n")
        f.write("通过这次作业，我把模拟数据和真实数据放在同一个回归分析框架下进行比较。")
        f.write("模拟数据适合理解模型是否能恢复已知机制，真实数据则更接近实际业务问题。")
        f.write("两者结合起来，可以帮助我更清楚地理解回归分析中的预测、推测、数据泄露和模型解释问题。\n")

    print(f"对比总结报告已保存：{summary_path}")


# ============================================================
# 7. 主函数：唯一执行入口
# ============================================================

def main():
    print("========== Week 11 作业开始运行 ==========")

    synthetic_metric_df, synthetic_coef_df, synthetic_vif = run_synthetic_workflow()

    kaggle_metric_df, kaggle_coef_df, kaggle_vif = run_kaggle_workflow()

    write_summary_report(synthetic_metric_df, kaggle_metric_df)

    print("\n========== Week 11 全部任务完成 ==========")
    print("请检查以下文件是否生成：")
    print(f"1. {SYNTHETIC_PATH}")
    print(f"2. {os.path.join(RESULTS_DIR, 'synthetic_report.md')}")
    print(f"3. {os.path.join(RESULTS_DIR, 'kaggle_report.md')}")
    print(f"4. {os.path.join(RESULTS_DIR, 'summary_comparison.md')}")


if __name__ == "__main__":
    main()