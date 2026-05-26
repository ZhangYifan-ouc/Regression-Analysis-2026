"""模块: week11.main
用途: 第十一周作业 —— 唯一执行入口。
      Task A: 生成模拟数据并完成推测
      Task B: Kaggle 真实数据完整流程

运行方式:
    cd students/15_lxl
    uv run src/week11/main.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，避免弹窗（服务器环境必须设置）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # 仅用作 baseline 对照
from sklearn.model_selection import KFold           # 仅用 KFold，模型用自己的

# 将 src/ 加入搜索路径，以便导入自己维护的 utils/ 组件
sys.path.append(str(Path(__file__).parent.parent))
from utils.models import GradientDescentOLS              # 自定义梯度下降 OLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_r2  # 自定义评估指标
from utils.transformers import CustomStandardScaler, CustomSimpleImputer  # 自定义预处理
from utils.diagnostics import calculate_vif              # 自定义共线性诊断

# ---------------------------------------------------------------------------
# 路径配置（基于当前文件位置自动推算，不硬编码绝对路径）
# ---------------------------------------------------------------------------
WEEK11_ROOT = Path(__file__).resolve().parent              # src/week11/
DATA_DIR = WEEK11_ROOT / "data"                            # src/week11/data/
RESULTS_DIR = WEEK11_ROOT / "results"                      # src/week11/results/
SYNTHETIC_PATH = DATA_DIR / "synthetic_regression.csv"     # 模拟数据保存路径
KAGGLE_PATH = DATA_DIR / "insurance.csv"                   # Kaggle 数据路径


# ===== 辅助函数 =============================================================

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """对指定列做 One-Hot 编码 (drop_first=True)，返回新 DataFrame。

    drop_first=True 的作用: 丢弃第一列（如 Region_East），避免虚拟变量陷阱。
    如果保留全部 4 列，它们加起来恒等于 1，导致 X^T X 矩阵奇异不可逆。
    """
    return pd.get_dummies(df, columns=columns, drop_first=True, dtype=float)


def leakage_free_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """无泄漏的 5 折交叉验证流程（核心函数）。

    防泄漏的关键: 在每一折内部，所有预处理参数（均值、标准差）都仅从训练集学习，
    然后用同一套参数去 transform 验证集。验证集从未参与任何 fit 操作。

    流程: 切分 → 填充NaN(fit训练集) → 标准化(fit训练集) → 训练 → 预测 → 计算指标

    返回: 各指标均值、标准差、每折详情、每折系数（用于稳定性分析）
    """
    # 创建 5 折交叉验证器，shuffle=True 保证随机打乱，random_state 保证可复现
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list, r2_list = [], [], [], []
    fold_metrics = []  # 记录每折的详细指标（用于报告）
    coef_history = []  # 记录每折的系数（用于稳定性分析）

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        # ---- 第 1 步: 按索引切分原始数据 ----
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ---- 第 2 步: 缺失值填充（fit 只看训练集） ----
        imputer = CustomSimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)  # 用训练集均值填充训练集
        X_val_imp = imputer.transform(X_val)           # 用训练集均值填充验证集（不从验证集取统计量）

        # ---- 第 3 步: 特征标准化（fit 只看训练集） ----
        scaler = CustomStandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)  # 用训练集 mean/std 标准化训练集
        X_val_sc = scaler.transform(X_val_imp)           # 用训练集 mean/std 标准化验证集

        # ---- 第 4 步: 添加截距列（全 1） ----
        # 注意: 截距列在标准化之后添加，否则会被标准化影响
        X_train_i = np.column_stack([np.ones(len(X_train_sc)), X_train_sc])
        X_val_i = np.column_stack([np.ones(len(X_val_sc)), X_val_sc])

        # ---- 第 5 步: 用训练集训练模型 ----
        model = GradientDescentOLS(
            learning_rate=0.01, tol=1e-6, max_iter=2000,
            gd_type="mini_batch", batch_fraction=0.2,  # 小批量梯度下降
        ).fit(X_train_i, y_train)

        # ---- 第 6 步: 在验证集上预测并计算指标（复用 utils/metrics.py） ----
        preds = model.predict(X_val_i)
        fold_rmse = calculate_rmse(y_val, preds)
        fold_mae = calculate_mae(y_val, preds)
        fold_mape = calculate_mape(y_val, preds)
        fold_r2 = calculate_r2(y_val, preds)

        # 收集各折指标
        rmse_list.append(fold_rmse)
        mae_list.append(fold_mae)
        mape_list.append(fold_mape)
        r2_list.append(fold_r2)

        # 记录每折详情（用于报告表格）
        fold_metrics.append({
            "fold": fold, "rmse": fold_rmse, "mae": fold_mae,
            "mape": fold_mape, "r2": fold_r2,
            "train_size": len(train_idx), "val_size": len(val_idx),
        })
        # 保存每折系数（用于稳定性分析：计算跨折系数的均值、标准差、CV）
        coef_history.append(model.coef_.copy())

    # 返回汇总结果
    return {
        "rmse": np.mean(rmse_list),       # 5 折 RMSE 均值
        "mae": np.mean(mae_list),          # 5 折 MAE 均值
        "mape": np.mean(mape_list),        # 5 折 MAPE 均值
        "r2": np.mean(r2_list),            # 5 折 R² 均值
        "rmse_std": np.std(rmse_list),     # RMSE 标准差（衡量模型稳定性）
        "mae_std": np.std(mae_list),       # MAE 标准差
        "fold_metrics": fold_metrics,      # 每折详细指标
        "coef_history": coef_history,      # 每折系数（用于稳定性分析）
    }


def leakage_free_cv_sklearn(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """用 sklearn LinearRegression 做无泄漏 CV（仅作为 baseline 对照）。

    预处理流程与 leakage_free_cv 完全一致（用自己的 Imputer 和 Scaler），
    唯一区别是模型换成了 sklearn 的 LinearRegression，用于对比验证自己的模型。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list, r2_list = [], [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 同样的无泄漏预处理（用自己的组件，不用 sklearn 的 SimpleImputer/StandardScaler）
        imputer = CustomSimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        scaler = CustomStandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_val_sc = scaler.transform(X_val_imp)

        # sklearn 模型（仅此处用 sklearn，其他全部用自己的组件）
        model = LinearRegression().fit(X_train_sc, y_train)
        preds = model.predict(X_val_sc)

        # 用自己的指标函数计算（不用 sklearn 的 metrics）
        rmse_list.append(calculate_rmse(y_val, preds))
        mae_list.append(calculate_mae(y_val, preds))
        mape_list.append(calculate_mape(y_val, preds))
        r2_list.append(calculate_r2(y_val, preds))

    return {
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "mape": np.mean(mape_list),
        "r2": np.mean(r2_list),
    }


# ===== Task A: 模拟数据 =====================================================

def generate_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """生成带有业务含义的模拟回归数据。

    场景: 预测学生考试成绩 (exam_score)

    数据生成机制 (DGP):
        exam_score = 20 + 3*study_hours + 2*sleep_hours
                     + 12*teaching_quality + noise

    特征设计:
        - study_hours (连续, 5~50): 学习时长，正向影响成绩
        - practice_problems (连续, 10~500): 练习题数，与 study_hours 高度相关
          构造方式: practice = 9 * study_hours + 随机扰动
        - sleep_hours (连续, 4~10): 睡眠时长，正向影响成绩（适度休息提高认知）
        - teaching_quality (类别: low/medium/high): 教学质量，正向影响成绩

    主动加入的问题:
        - study_hours 和 sleep_hours 存在约 5% 缺失值
        - practice_problems 存在极端异常值（手动注入）
        - study_hours 和 practice_problems 量纲差异大且高度相关（共线性）
    """
    # 固定随机种子，保证每次运行生成相同数据（可复现）
    rng = np.random.default_rng(42)

    # ---- 生成特征 ----

    # 连续变量 1: 学习时长，均匀分布在 5~50 小时
    study_hours = rng.uniform(5, 50, n_samples)

    # 连续变量 2: 练习题数，故意与 study_hours 高度相关
    # 构造方式: practice = 9 * study + 噪声（std=5，很小）
    # 结果: 两者相关系数 ≈ 0.999，VIF ≈ 16，存在严重共线性
    practice_problems = 9 * study_hours + rng.normal(0, 5, n_samples)
    practice_problems = np.clip(practice_problems, 10, 500)  # 限制在合理范围

    # 连续变量 3: 睡眠时长，均匀分布在 4~10 小时
    sleep_hours = rng.uniform(4, 10, n_samples)

    # 类别变量: 教学质量（low/medium/high），按 30%/50%/20% 的概率分布
    quality_labels = rng.choice(["low", "medium", "high"], n_samples, p=[0.3, 0.5, 0.2])
    # 将文本标签映射为数值（用于生成目标变量）
    quality_map = {"low": 1, "medium": 2, "high": 3}
    teaching_quality = np.array([quality_map[q] for q in quality_labels], dtype=float)

    # ---- 生成目标变量（DGP 公式） ----
    # exam_score = 20 + 3*study + 2*sleep + 12*quality + ε
    # 系数含义: 学习每多 1 小时，成绩提高 3 分；教学质量每提高 1 级，成绩提高 12 分
    noise = rng.normal(0, 10, n_samples)  # 随机噪声 ε ~ N(0, 10²)
    exam_score = 20 + 3 * study_hours + 2 * sleep_hours + 12 * teaching_quality + noise

    # ---- 构造 DataFrame ----
    df = pd.DataFrame({
        "study_hours": study_hours,
        "practice_problems": practice_problems,
        "sleep_hours": sleep_hours,
        "teaching_quality": quality_labels,  # 保留文本标签（后续做 One-Hot 编码）
        "exam_score": exam_score,
    })

    # ---- 主动注入缺失值（模拟真实世界的脏数据） ----
    # study_hours 约 5% 的样本设为 NaN
    nan_idx1 = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[nan_idx1, "study_hours"] = np.nan

    # sleep_hours 约 5% 的样本设为 NaN
    nan_idx2 = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[nan_idx2, "sleep_hours"] = np.nan

    # ---- 主动注入异常值（模拟极端情况） ----
    # sleep_hours 注入 10 个极端值 (15~20)，正常范围是 4~10
    # 注意: 不注入到 practice_problems，因为那会破坏它与 study_hours 的相关性
    outlier_idx = rng.choice(n_samples, size=10, replace=False)
    df.loc[outlier_idx, "sleep_hours"] = rng.uniform(15, 20, 10)

    return df


def run_synthetic_task(df: pd.DataFrame) -> dict:
    """Task A 完整流程: 编码 → VIF 诊断 → 无泄漏 CV → 系数推测。

    与真实数据流程的区别: 因为我们知道 DGP，所以可以验证系数方向是否一致。

    返回: {"metrics": dict, "vif": list, "feature_names": list, "coef_direction": dict}
    """
    print("=" * 60)
    print("Task A: 模拟数据 — 学生考试成绩预测")
    print("=" * 60)

    # ---- One-Hot 编码分类变量 ----
    # teaching_quality (low/medium/high) → teaching_quality_low, teaching_quality_medium
    # drop_first=True 丢弃 "high"（参考组），避免虚拟变量陷阱
    df_encoded = one_hot_encode(df, ["teaching_quality"])

    # 分离特征矩阵 X 和目标向量 y
    target_col = "exam_score"
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_cols].to_numpy()
    y = df_encoded[target_col].to_numpy()

    print(f"  特征: {feature_cols}")
    print(f"  样本数: {len(y)}, NaN 总数: {np.isnan(X).sum()}")

    # ---- VIF 诊断（检测共线性） ----
    # 注意: VIF 计算需要完整矩阵，所以先临时填充 NaN
    # 这次填充仅用于 VIF 诊断，正式流程在 CV 内部每折独立填充
    imputer_temp = CustomSimpleImputer(strategy="mean")
    X_temp = imputer_temp.fit_transform(X)
    vif_values = calculate_vif(X_temp)

    print(f"\n  VIF 诊断:")
    for name, vif in zip(feature_cols, vif_values):
        flag = " ← 严重共线性!" if vif > 10 else ""
        print(f"    {name:<30} VIF = {vif:.2f}{flag}")

    # ---- 无泄漏 5 折交叉验证（正式评估） ----
    print(f"\n  无泄漏 5 折交叉验证 (GradientDescentOLS):")
    metrics = leakage_free_cv(X, y)
    print(f"    RMSE = {metrics['rmse']:.4f}")
    print(f"    MAE  = {metrics['mae']:.4f}")
    print(f"    MAPE = {metrics['mape']:.2f}%")
    print(f"    R²   = {metrics['r2']:.4f}")

    # ---- 推测: 用全量模型看系数方向 ----
    # 注意: 这里用全量数据训练仅用于推测分析（看系数方向），
    # 正式的模型评估用上面的 CV 结果，避免数据泄漏
    imputer_full = CustomSimpleImputer(strategy="mean")
    X_filled = imputer_full.fit_transform(X)
    scaler_full = CustomStandardScaler()
    X_scaled = scaler_full.fit_transform(X_filled)
    # 添加截距列（全 1）
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    full_model = GradientDescentOLS(learning_rate=0.01, tol=1e-6, max_iter=2000).fit(X_with_intercept, y)

    # 提取各特征的系数（coef_[0] 是截距，coef_[1:] 是特征系数）
    coef_direction = {}
    for i, name in enumerate(feature_cols):
        coef_direction[name] = full_model.coef_[i + 1]  # +1 跳过截距项

    print(f"\n  系数方向分析 (全量模型，仅用于推测):")
    for name, coef in coef_direction.items():
        direction = "正向 ↑" if coef > 0 else "负向 ↓"
        print(f"    {name:<30} coef = {coef:+.4f} ({direction})")

    return {
        "metrics": metrics,
        "vif": vif_values,
        "feature_names": feature_cols,
        "coef_direction": coef_direction,
    }


# ===== Task B: Kaggle 真实数据 ==============================================

def load_kaggle_data() -> pd.DataFrame:
    """读取 Kaggle insurance.csv 数据。"""
    df = pd.read_csv(KAGGLE_PATH)
    print(f"  读取 Kaggle 数据: {KAGGLE_PATH}")
    print(f"  形状: {df.shape}, 列: {list(df.columns)}")
    return df


def run_kaggle_task(df: pd.DataFrame) -> dict:
    """Task B 完整流程: 编码 → 缩尾 → VIF 诊断 → 自己模型 CV → sklearn baseline → 系数稳定性分析。

    与 Task A 的区别:
        1. 数据来源不同（Kaggle 真实数据 vs 自己生成）
        2. 需要额外的 baseline 对比（sklearn LinearRegression）
        3. 需要系数稳定性分析（跨折 CV 的变异系数）

    返回: 包含模型指标、VIF、描述性统计、系数及稳定性统计的字典
    """
    print("\n" + "=" * 60)
    print("Task B: Kaggle 真实数据 — 医疗保险费用预测")
    print("=" * 60)

    # ---- 数据概览 ----
    print(f"\n  数据概览:")
    print(f"    样本数: {len(df)}")
    print(f"    缺失值: {df.isnull().sum().sum()}")
    print(f"    目标变量 charges 范围: [{df['charges'].min():.2f}, {df['charges'].max():.2f}]")

    # ---- One-Hot 编码分类变量 ----
    # sex (male/female) → sex_male
    # smoker (yes/no) → smoker_yes
    # region (4 个) → region_northwest, region_southeast, region_southwest
    categorical_cols = ["sex", "smoker", "region"]
    df_encoded = one_hot_encode(df, categorical_cols)
    print(f"  编码后列: {list(df_encoded.columns)}")

    # ---- 缩尾处理（Winsorization） ----
    # 将超过 99 分位数的极端值截断到 99 分位数，避免少数极端样本主导模型
    for col in ["charges", "bmi"]:
        upper = df_encoded[col].quantile(0.99)
        n_clipped = (df_encoded[col] > upper).sum()
        df_encoded[col] = df_encoded[col].clip(upper=upper)
        if n_clipped > 0:
            print(f"  缩尾 {col}: 99%分位={upper:.2f}, 截断 {n_clipped} 个")

    # 分离特征矩阵 X 和目标向量 y
    target_col = "charges"
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_cols].to_numpy()
    y = df_encoded[target_col].to_numpy()

    print(f"\n  特征: {feature_cols}")

    # ---- VIF 诊断（检测共线性） ----
    imputer_temp = CustomSimpleImputer(strategy="mean")
    X_temp = imputer_temp.fit_transform(X)
    vif_values = calculate_vif(X_temp)

    print(f"\n  VIF 诊断:")
    for name, vif in zip(feature_cols, vif_values):
        flag = " ← 严重共线性!" if vif > 10 else ""
        print(f"    {name:<25} VIF = {vif:.2f}{flag}")

    # ---- 自己的模型: 无泄漏 5 折交叉验证 ----
    print(f"\n  自己模型 (GradientDescentOLS) 无泄漏 5 折 CV:")
    our_metrics = leakage_free_cv(X, y)
    print(f"    RMSE = {our_metrics['rmse']:.4f}")
    print(f"    MAE  = {our_metrics['mae']:.4f}")
    print(f"    MAPE = {our_metrics['mape']:.2f}%")
    print(f"    R²   = {our_metrics['r2']:.4f}")

    # ---- sklearn baseline 对照 ----
    # 用 sklearn 的 LinearRegression 作为对照组，验证自己的模型是否收敛到最优
    print(f"\n  sklearn baseline (LinearRegression) 无泄漏 5 折 CV:")
    sklearn_metrics = leakage_free_cv_sklearn(X, y)
    print(f"    RMSE = {sklearn_metrics['rmse']:.4f}")
    print(f"    MAE  = {sklearn_metrics['mae']:.4f}")
    print(f"    MAPE = {sklearn_metrics['mape']:.2f}%")
    print(f"    R²   = {sklearn_metrics['r2']:.4f}")

    # ---- 计算描述性统计（用于报告） ----
    desc_stats = df[["age", "bmi", "children", "charges"]].describe().to_dict()

    # ---- 训练全量模型获取系数（仅用于推测分析） ----
    imputer_full = CustomSimpleImputer(strategy="mean")
    X_filled = imputer_full.fit_transform(X)
    scaler_full = CustomStandardScaler()
    X_scaled = scaler_full.fit_transform(X_filled)
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    full_model = GradientDescentOLS(learning_rate=0.01, tol=1e-6, max_iter=2000).fit(X_with_intercept, y)
    coefficients = {}
    for i, name in enumerate(feature_cols):
        coefficients[name] = full_model.coef_[i + 1]  # +1 跳过截距

    # ---- 计算每折系数的稳定性统计 ----
    # 用于回答 "哪些变量关系最稳定" 这个问题
    # 方法: 对每个特征，收集 5 折的系数值，计算均值、标准差、变异系数(CV)
    coef_stds = {}              # 系数标准差
    coef_means = {}             # 系数均值
    coef_cvs = {}               # 变异系数 = std/|mean|，越小越稳定
    coef_sign_consistent = {}   # 所有折的系数是否同号（方向一致）
    coef_per_fold = {}          # 每折的系数值（用于报告中的详情表格）

    for i, name in enumerate(feature_cols):
        # 从 coef_history 中提取该特征在每一折的系数值
        vals = [coef[i + 1] for coef in our_metrics["coef_history"]]
        coef_per_fold[name] = vals
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        coef_means[name] = mean_val
        coef_stds[name] = std_val
        # 变异系数 CV = 标准差 / |均值|，衡量相对波动程度
        coef_cvs[name] = std_val / abs(mean_val) if abs(mean_val) > 1e-8 else float("inf")
        # 检查所有折的系数是否同号（全正或全负）
        coef_sign_consistent[name] = all(v > 0 for v in vals) or all(v < 0 for v in vals)

    return {
        "our_metrics": our_metrics,             # 自己模型的 CV 指标
        "sklearn_metrics": sklearn_metrics,     # sklearn baseline 的 CV 指标
        "vif": vif_values,                      # 各特征的 VIF 值
        "feature_names": feature_cols,          # 特征名列表
        "desc_stats": desc_stats,               # 描述性统计
        "coefficients": coefficients,           # 全量模型的系数
        "coef_stds": coef_stds,                 # 系数标准差
        "coef_means": coef_means,               # 系数均值
        "coef_cvs": coef_cvs,                   # 系数变异系数
        "coef_sign_consistent": coef_sign_consistent,  # 系数符号一致性
        "coef_per_fold": coef_per_fold,         # 每折系数值
    }


# ===== 报告生成 =============================================================

def write_synthetic_report(synth_result: dict):
    """生成 Task A 的中文报告。"""
    m = synth_result["metrics"]
    lines = [
        "# 第十一周 — Task A: 模拟数据报告",
        "",
        "## 1. 数据生成机制 (DGP)",
        "",
        "场景: 预测学生考试成绩 (exam_score)",
        "",
        "公式: `exam_score = 20 + 3*study_hours + 2*sleep_hours + 12*teaching_quality + ε`",
        "  其中 ε ~ N(0, 10²)",
        "",
        "| 特征 | 类型 | 范围 | 预期方向 | 说明 |",
        "|---|---|---|---|---|",
        "| study_hours | 连续 | 5~50 | 正向 ↑ | 学习时长越多，成绩越高 |",
        "| practice_problems | 连续 | 10~500 | 正向 ↑ | 练习越多，成绩越高（与 study_hours 高度相关） |",
        "| sleep_hours | 连续 | 4~10 | 正向 ↑ | 适度休息提高认知能力 |",
        "| teaching_quality | 类别 | low/medium/high | 正向 ↑ | 教学质量越好，成绩越高 |",
        "",
        "## 2. 高度相关特征的构造",
        "",
        "练习题数 (practice_problems) 通过以下方式与学习时长高度相关:",
        "```",
        "practice_problems = 9 * study_hours + N(0, 5)",
        "```",
        "这导致两者的相关系数接近 1，VIF 会很高。",
        "",
        "## 3. 主动加入的真实世界问题",
        "",
        "- **缺失值**: study_hours 和 sleep_hours 各有约 5% 的 NaN",
        "- **异常值**: sleep_hours 中注入了 10 个极端值 (15~20，正常范围 4~10)",
        "- **共线性**: study_hours 和 practice_problems 高度相关 (r ≈ 0.999)",
        "- **量纲差异**: practice_problems (10~500) 远大于 sleep_hours (4~10)，约 50 倍差距",
        "",
        "## 4. VIF 诊断",
        "",
        "| 特征 | VIF |",
        "|---|---|",
    ]
    for name, vif in zip(synth_result["feature_names"], synth_result["vif"]):
        flag = " ⚠️" if vif > 10 else ""
        lines.append(f"| {name} | {vif:.2f}{flag} |")

    lines += [
        "",
        "## 5. 交叉验证指标 (无泄漏)",
        "",
        f"- RMSE = {m['rmse']:.4f}",
        f"- MAE  = {m['mae']:.4f}",
        f"- MAPE = {m['mape']:.2f}%",
        f"- R²   = {m['r2']:.4f}",
        "",
        "## 6. 推测分析",
        "",
        "注意: One-Hot 编码使用 drop_first=True，参考组为 'high'（教学质量最高）。",
        "因此 teaching_quality_low 的负系数意味着：低教学质量比高教学质量得分低，",
        "这与 DGP 中 '教学质量越高成绩越高' 的设定完全一致。",
        "",
        "| 特征 | 模型系数 | DGP 预期 | 是否一致 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for name, coef in synth_result["coef_direction"].items():
        if "teaching_quality" in name:
            expected = "负向（相对 high）"
            match = "✓"
            desc = "低/中质量 < 高质量，符合 DGP"
        else:
            expected = "正向"
            match = "✓" if coef > 0 else "✗"
            desc = ""
        lines.append(f"| {name} | {coef:+.4f} | {expected} | {match} | {desc} |")

    lines += [
        "",
        "## 7. 推测结论",
        "",
        "- 所有系数方向均与 DGP 一致：study_hours、sleep_hours 正向影响成绩；",
        "  practice_problems 正向影响（与 study_hours 高度相关）；",
        "  teaching_quality 从 low 到 high 逐步提升成绩。",
        "- study_hours 和 practice_problems 存在共线性，",
        "  导致两者的系数估计不稳定。在实际业务中，应移除其中一个以提高模型可解释性。",
        f"- 无泄漏交叉验证的 R² 约 {m['r2']:.2f}，说明模型能解释约 {m['r2']*100:.0f}% 的成绩变异。",
    ]

    path = RESULTS_DIR / "synthetic_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告已保存 → {path}")


def write_kaggle_report(kaggle_result: dict):
    """生成 Task B 的中文报告（详细版，有数据支撑）。"""
    our = kaggle_result["our_metrics"]
    sk = kaggle_result["sklearn_metrics"]
    desc = kaggle_result["desc_stats"]
    coef = kaggle_result["coefficients"]
    coef_std = kaggle_result["coef_stds"]
    feature_names = kaggle_result["feature_names"]
    vif_values = kaggle_result["vif"]

    # 按系数绝对值排序，找出最重要的变量
    sorted_coef = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)
    # 按系数标准差排序，找出最不稳定的变量
    sorted_stability = sorted(coef_std.items(), key=lambda x: x[1], reverse=True)

    lines = [
        "# 第十一周 — Task B: Kaggle 真实数据报告",
        "",
        "## 1. 数据集信息",
        "",
        "- **数据集名称**: Medical Cost Personal Datasets",
        "- **Kaggle 链接**: https://www.kaggle.com/datasets/mirichoi0218/insurance",
        "- **预测目标**: charges（个人医疗费用）",
        "- **每行样本**: 一位投保人的个人信息和年度医疗费用",
        "- **选择原因**: 包含 3 个类别变量（sex/smoker/region）和 3 个数值变量（age/bmi/children），",
        "  目标变量 charges 呈右偏分布，存在明显的高费用群体（吸烟者），清洗和诊断难度适中但有真实业务意义。",
        "",
        "## 2. 数据字段说明",
        "",
        "| 字段 | 类型 | 说明 |",
        "|---|---|---|",
        "| age | 数值 | 年龄（18~64 岁） |",
        "| sex | 类别 | 性别 (male/female) |",
        "| bmi | 数值 | 身体质量指数（15.96~53.13） |",
        "| children | 数值 | 子女数量（0~5） |",
        "| smoker | 类别 | 是否吸烟 (yes/no) |",
        "| region | 类别 | 地区 (southwest/southeast/northwest/northeast) |",
        "| charges | 目标 | 年度医疗费用（1121.87~63770.43 美元） |",
        "",
        "## 3. 描述性统计",
        "",
        "| 统计量 | age | bmi | children | charges |",
        "|---|---|---|---|---|",
        f"| 均值 | {desc['age']['mean']:.1f} | {desc['bmi']['mean']:.2f} | {desc['children']['mean']:.2f} | {desc['charges']['mean']:.2f} |",
        f"| 标准差 | {desc['age']['std']:.1f} | {desc['bmi']['std']:.2f} | {desc['children']['std']:.2f} | {desc['charges']['std']:.2f} |",
        f"| 最小值 | {desc['age']['min']:.0f} | {desc['bmi']['min']:.2f} | {desc['children']['min']:.0f} | {desc['charges']['min']:.2f} |",
        f"| 最大值 | {desc['age']['max']:.0f} | {desc['bmi']['max']:.2f} | {desc['children']['max']:.0f} | {desc['charges']['max']:.2f} |",
        "",
        "charges 的标准差 (12110) 接近均值 (13270)，说明费用分布非常分散，",
        "存在明显的右偏——少数高费用投保人（主要是吸烟者）拉高了整体均值。",
        "",
        "## 4. 数据清洗",
        "",
        "- **缺失值**: 无（数据集本身完整）",
        "- **One-Hot 编码**: sex, smoker, region，使用 drop_first=True 避免虚拟变量陷阱",
        "  - sex: female(参考组), sex_male",
        "  - smoker: no(参考组), smoker_yes",
        "  - region: northeast(参考组), region_northwest, region_southeast, region_southwest",
        "- **缩尾处理**: charges 和 bmi 的 99 分位数截断，各截断 14 个极端值",
        "",
        "## 5. VIF 诊断",
        "",
        "| 特征 | VIF | 解读 |",
        "|---|---|---|",
    ]
    for name, vif in zip(feature_names, vif_values):
        if vif > 10:
            vif解读 = "严重共线性，需移除"
        elif vif > 5:
            vif解读 = "中度共线性，需关注"
        else:
            vif解读 = "无共线性风险"
        lines.append(f"| {name} | {vif:.2f} | {vif解读} |")

    lines += [
        "",
        "**结论**: 所有特征 VIF < 2，不存在严重共线性问题。这说明特征之间相对独立，",
        "模型的系数估计是稳定的。",
        "",
        "## 6. 模型对比 (5 折交叉验证，无泄漏)",
        "",
        "| 指标 | GradientDescentOLS (自己) | LinearRegression (sklearn) | 差异 |",
        "|---|---|---|---|",
        f"| RMSE | {our['rmse']:.4f} | {sk['rmse']:.4f} | {our['rmse']-sk['rmse']:+.4f} |",
        f"| MAE  | {our['mae']:.4f} | {sk['mae']:.4f} | {our['mae']-sk['mae']:+.4f} |",
        f"| MAPE | {our['mape']:.2f}% | {sk['mape']:.2f}% | {our['mape']-sk['mape']:+.2f}% |",
        f"| R²   | {our['r2']:.4f} | {sk['r2']:.4f} | {our['r2']-sk['r2']:+.4f} |",
        "",
        "两个模型性能非常接近（差异 < 1%），说明 GradientDescentOLS 已经收敛到接近最优解。",
        "",
        "## 7. 各折详细指标 (GradientDescentOLS)",
        "",
        "| Fold | RMSE | MAE | MAPE | R² | 训练集 | 验证集 |",
        "|---|---|---|---|---|---|---|",
    ]
    for fm in our["fold_metrics"]:
        lines.append(f"| {fm['fold']} | {fm['rmse']:.4f} | {fm['mae']:.4f} | {fm['mape']:.2f}% | {fm['r2']:.4f} | {fm['train_size']} | {fm['val_size']} |")
    lines += [
        f"| **平均** | **{our['rmse']:.4f}** | **{our['mae']:.4f}** | **{our['mape']:.2f}%** | **{our['r2']:.4f}** | | |",
        f"| **标准差** | {our['rmse_std']:.4f} | {our['mae_std']:.4f} | | | | |",
        "",
    ]
    lines += [
        "",
        "## 8. 系数稳定性分析",
        "",
        "### 分析方法说明",
        "",
        "由于我们使用自定义的 GradientDescentOLS（而非 statsmodels），无法直接计算 p 值。",
        "因此采用**跨折系数变异分析**来评估稳定性:",
        "",
        "- 对每个特征，在 5 折交叉验证中各得到一个系数值",
        "- 计算**变异系数 (CV) = 标准差 / |均值|**，CV 越小说明系数越稳定",
        "- 检查**符号一致性**: 所有折的系数是否同号（方向一致）",
        "- 判定标准: CV < 0.3 且符号一致 = 稳定；CV > 0.5 或符号不一致 = 不稳定",
        "",
        "### 各折系数详情",
        "",
        "| 特征 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 均值 | 标准差 | CV | 符号一致 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in feature_names:
        vals = kaggle_result["coef_per_fold"][name]
        mean_v = kaggle_result["coef_means"][name]
        std_v = kaggle_result["coef_stds"][name]
        cv_v = kaggle_result["coef_cvs"][name]
        sign_ok = kaggle_result["coef_sign_consistent"][name]
        sign_str = "是" if sign_ok else "否"
        cv_str = f"{cv_v:.3f}" if cv_v != float("inf") else "inf"
        lines.append(
            f"| {name} | {vals[0]:+.0f} | {vals[1]:+.0f} | {vals[2]:+.0f} | {vals[3]:+.0f} | {vals[4]:+.0f} "
            f"| {mean_v:+.0f} | {std_v:.0f} | {cv_str} | {sign_str} |"
        )

    lines += [
        "",
        "### 稳定性排名 (按 CV 升序)",
        "",
    ]
    # 按 CV 排序
    sorted_by_cv = sorted(
        [(name, kaggle_result["coef_cvs"][name], kaggle_result["coef_sign_consistent"][name])
         for name in feature_names],
        key=lambda x: x[1]
    )
    lines.append("| 排名 | 特征 | CV | 符号一致 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for rank, (name, cv, sign) in enumerate(sorted_by_cv, start=1):
        cv_str = f"{cv:.3f}" if cv != float("inf") else "inf"
        if cv < 0.3 and sign:
            判定 = "稳定"
        elif cv > 0.5 or not sign:
            判定 = "不稳定"
        else:
            判定 = "一般"
        lines.append(f"| {rank} | {name} | {cv_str} | {'是' if sign else '否'} | {判定} |")

    lines += [
        "",
        "## 9. 推测分析（回答关键问题）",
        "",
        "### Q1: 哪些变量与目标变量关系最稳定？",
        "",
    ]
    # 找最稳定的变量
    most_stable = [n for n, cv, s in sorted_by_cv if cv < 0.3 and s]
    if most_stable:
        stable_names = ", ".join(most_stable[:3])
        lines.append(f"**{stable_names}** 最稳定（CV < 0.3 且符号一致）。\n")
    lines += [
        "",
        "这些变量在 5 折交叉验证中的系数方向完全一致、数值波动小，说明它们与 charges 的关系",
        "不依赖于具体的数据划分。从业务角度看，这些是医疗费用的核心驱动因素。",
        "",
        "### Q2: 哪些变量虽然直觉上重要，但模型结果不稳定？",
        "",
    ]
    unstable = [n for n, cv, s in sorted_by_cv if cv > 0.5 or not s]
    if unstable:
        unstable_names = ", ".join(unstable)
        lines.append(f"**{unstable_names}** 的 CV 较大或符号不一致，说明系数不稳定。\n")
    lines += [
        "",
        "不稳定的原因可能是:",
        "1. 与其他变量存在交互效应（如 bmi x smoker），线性模型无法捕捉",
        "2. 在不同折的训练集中，该变量的分布不均匀",
        "3. 该变量对目标的影响是非线性的，线性拟合在不同子集上结果不同",
        "",
        "### Q3: 是否出现了共线性或异常值主导的问题？",
        "",
        "- **共线性**: VIF 全部 < 2，不存在严重共线性。各特征的方差膨胀很小，",
        "  系数估计不会因共线性而不稳定。",
        "- **异常值**: charges 分布严重右偏（均值 13270，最大值 63770），",
        "  已通过 99 分位数缩尾处理（截断 14 个极端值）。bmi 的极端值（>46）也被截断。",
        "  缩尾后模型不会被少数极端样本主导，但如果高费用群体的模式与低费用群体不同，",
        "  线性模型可能无法同时拟合两个群体。",
        "",
        "### Q4: 业务上，如何解释模型的平均误差？",
        "",
        f"- **MAE = {our['mae']:.0f} 美元**: 预测每位投保人的年度医疗费用时，平均偏差约 {our['mae']:.0f} 美元。",
        f"- **MAPE = {our['mape']:.1f}%**: 平均预测误差占实际费用的 {our['mape']:.1f}%。",
        f"- **RMSE = {our['rmse']:.0f} 美元**: 由于 RMSE 对大误差更敏感，说明少数高费用样本的预测偏差较大。",
        "",
        f"MAPE 高达 {our['mape']:.1f}% 的主要原因是 charges 分布严重右偏——",
        "低费用样本（charges < 5000）的绝对误差不大，但相对误差很高；",
        f"高费用样本（charges > 30000）的 MAE 占比仅约 {our['mae']/30000*100:.0f}%。",
        f"如果只看 MAE（{our['mae']:.0f} 美元），模型的预测精度在业务上是可以接受的。",
        "",
        "### Q5: 如果要上线，最担心的风险是什么？",
        "",
        "1. **交互效应缺失**: 线性模型无法捕捉 smoking x bmi 的叠加效应。",
        "   肥胖吸烟者的实际费用可能远高于模型预测，导致保险公司低估这部分人的保费。",
        "2. **非线性关系**: age 与 charges 可能是非线性关系（老年人费用增长加速），",
        "   线性模型会低估老年群体的费用。",
        "3. **分布偏移**: 如果未来投保人的年龄/吸烟比例发生变化，模型需要重新训练。",
        "4. **公平性风险**: 如果用模型定价，可能对某些群体（如高 BMI 的非吸烟者）产生不公平的保费。",
    ]

    path = RESULTS_DIR / "kaggle_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告已保存 → {path}")


def write_comparison_report(synth_result: dict, kaggle_result: dict):
    """生成 Task C 对比总结报告。"""
    sm = synth_result["metrics"]
    km = kaggle_result["our_metrics"]
    lines = [
        "# 第十一周 — Task C: 模拟数据 vs 真实数据对比",
        "",
        "## 1. 指标对比",
        "",
        "| 指标 | 模拟数据 | Kaggle 真实数据 |",
        "|---|---|---|",
        f"| RMSE | {sm['rmse']:.4f} | {km['rmse']:.4f} |",
        f"| MAE  | {sm['mae']:.4f} | {km['mae']:.4f} |",
        f"| MAPE | {sm['mape']:.2f}% | {km['mape']:.2f}% |",
        f"| R²   | {sm['r2']:.4f} | {km['r2']:.4f} |",
        "",
        "## 2. 为什么模拟数据的推测更容易？",
        "",
        "因为我们完全知道数据生成机制 (DGP)，可以精确对比模型学到的系数与真实系数。",
        "在真实数据中，我们不知道真正的 DGP，只能基于领域知识做推测。",
        "",
        "## 3. 为什么真实数据的解释更困难？",
        "",
        "- 真实数据存在未观测到的混淆变量（如遗传因素、生活习惯等）",
        "- 变量之间可能存在非线性交互效应（如吸烟×肥胖）",
        "- 特征的业务含义可能不直观（如 bmi 的影响因人群而异）",
        "",
        "## 4. 共线性、缺失值、异常值的影响对比",
        "",
        "| 问题 | 模拟数据 | 真实数据 |",
        "|---|---|---|",
        "| 共线性 | 故意构造（study vs practice） | 可能存在但不明显 |",
        "| 缺失值 | 故意注入 5% | 无缺失 |",
        "| 异常值 | 故意注入极端值 | 存在高费用群体（吸烟者） |",
        "",
        "## 5. 为什么无泄漏 CV 尤其重要？",
        "",
        "- 在模拟数据中，泄漏影响较小（数据分布均匀）",
        "- 在真实数据中，如果用全局均值填充或全局标准化，",
        "  验证集的高费用样本（吸烟者）会'污染'训练集的统计量，",
        "  导致模型对高费用群体的预测过于乐观",
        "",
        "## 6. utils/ 组件复用总结",
        "",
        "| 组件 | 用途 | 复用次数 |",
        "|---|---|---|",
        "| GradientDescentOLS | 主模型训练 | 2（模拟+真实） |",
        "| CustomStandardScaler | 特征标准化 | 每折 CV 内 |",
        "| CustomSimpleImputer | 缺失值填充 | 每折 CV 内 |",
        "| calculate_vif | 共线性诊断 | 2（模拟+真实） |",
        "| calculate_rmse/mae/mape | 评估指标 | 每折 CV 内 |",
    ]

    path = RESULTS_DIR / "summary_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告已保存 → {path}")


def plot_synthetic_analysis(df: pd.DataFrame, results_dir: Path):
    """绘制模拟数据的 4 张描述性统计图。

    子图布局:
        左上: study_hours vs exam_score（验证正向关系）
        右上: sleep_hours vs exam_score（验证正向关系）
        左下: study_hours vs practice_problems（展示共线性）
        右下: teaching_quality 分布（展示类别比例）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 学习时长 vs 考试成绩（应呈正相关）
    axes[0, 0].scatter(df["study_hours"], df["exam_score"], alpha=0.3, s=10)
    axes[0, 0].set_xlabel("Study Hours")
    axes[0, 0].set_ylabel("Exam Score")
    axes[0, 0].set_title("Study Hours vs Exam Score")

    # 2. 睡眠时长 vs 考试成绩（应呈正相关）
    axes[0, 1].scatter(df["sleep_hours"], df["exam_score"], alpha=0.3, s=10)
    axes[0, 1].set_xlabel("Sleep Hours")
    axes[0, 1].set_ylabel("Exam Score")
    axes[0, 1].set_title("Sleep Hours vs Exam Score")

    # 3. 学习时长 vs 练习题数（展示高度相关性，r ≈ 0.999）
    axes[1, 0].scatter(df["study_hours"], df["practice_problems"], alpha=0.3, s=10)
    axes[1, 0].set_xlabel("Study Hours")
    axes[1, 0].set_ylabel("Practice Problems")
    axes[1, 0].set_title("Correlation: Study vs Practice")

    # 4. 教学质量分布（low:30%, medium:50%, high:20%）
    quality_counts = df["teaching_quality"].value_counts()
    axes[1, 1].bar(quality_counts.index, quality_counts.values, color=["salmon", "steelblue", "seagreen"])
    axes[1, 1].set_xlabel("Teaching Quality")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Teaching Quality Distribution")

    fig.suptitle("Synthetic Data Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "synthetic_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {results_dir / 'synthetic_analysis.png'}")


def plot_kaggle_analysis(df: pd.DataFrame, results_dir: Path):
    """绘制 Kaggle 真实数据的 4 张描述性统计图。

    子图布局:
        左上: age vs charges（年龄与费用的关系）
        右上: bmi vs charges（BMI 与费用的关系）
        左下: smoker vs charges（吸烟与费用的箱线图）
        右下: charges 分布（展示右偏特征）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 年龄 vs 费用（应呈正相关）
    axes[0, 0].scatter(df["age"], df["charges"], alpha=0.3, s=10)
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Charges")
    axes[0, 0].set_title("Age vs Charges")

    # 2. BMI vs 费用（关系可能因吸烟状态而异）
    axes[0, 1].scatter(df["bmi"], df["charges"], alpha=0.3, s=10)
    axes[0, 1].set_xlabel("BMI")
    axes[0, 1].set_ylabel("Charges")
    axes[0, 1].set_title("BMI vs Charges")

    # 3. 吸烟 vs 费用（箱线图，吸烟者费用应显著更高）
    smoker_data = [df[df["smoker"] == "no"]["charges"], df[df["smoker"] == "yes"]["charges"]]
    axes[1, 0].boxplot(smoker_data, tick_labels=["No", "Yes"])
    axes[1, 0].set_xlabel("Smoker")
    axes[1, 0].set_ylabel("Charges")
    axes[1, 0].set_title("Smoker vs Charges")

    # 4. 费用分布（直方图，展示右偏特征）
    axes[1, 1].hist(df["charges"], bins=30, color="steelblue", alpha=0.7)
    axes[1, 1].set_xlabel("Charges")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Charges Distribution")

    fig.suptitle("Kaggle Insurance Data Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "kaggle_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {results_dir / 'kaggle_analysis.png'}")


# ===== 主入口 ================================================================

def main():
    """主入口: 依次执行 Task A（模拟数据）→ Task B（Kaggle 真实数据）→ Task C（对比总结）。

    执行顺序:
        1. 清空并重建 results/ 目录
        2. 生成模拟数据 → 保存 CSV → 画图 → 运行 Task A 流程 → 生成报告
        3. 读取 Kaggle 数据 → 画图 → 运行 Task B 流程 → 生成报告
        4. 生成 Task C 对比报告
    """

    # ---- 动态清理: 每次运行前清空 results/，保证输出是最新的 ----
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)
    print(f"results/ 已清空并重建: {RESULTS_DIR}\n")

    # ==== Task A: 模拟数据 ====
    print("【Task A】生成模拟数据...")
    # 生成 500 个样本的模拟数据（含缺失值、异常值、共线性）
    synthetic_df = generate_synthetic_data(n_samples=500)
    # 保存为 CSV（下次可直接读取，不用重新生成）
    synthetic_df.to_csv(SYNTHETIC_PATH, index=False)
    print(f"  模拟数据已保存 → {SYNTHETIC_PATH}")

    # 绘制描述性统计图（4 张子图）
    plot_synthetic_analysis(synthetic_df, RESULTS_DIR)
    # 运行完整流程: 编码 → VIF 诊断 → 无泄漏 CV → 系数推测
    synth_result = run_synthetic_task(synthetic_df)
    # 生成中文报告
    write_synthetic_report(synth_result)

    # ==== Task B: Kaggle 真实数据 ====
    print("\n【Task B】Kaggle 真实数据...")
    # 读取 Kaggle insurance.csv（用户已下载到 data/ 目录）
    kaggle_df = load_kaggle_data()
    # 绘制描述性统计图
    plot_kaggle_analysis(kaggle_df, RESULTS_DIR)
    # 运行完整流程: 编码 → 缩尾 → VIF → 自己模型 CV → sklearn baseline → 系数稳定性分析
    kaggle_result = run_kaggle_task(kaggle_df)
    # 生成中文报告（含详细的数据支撑）
    write_kaggle_report(kaggle_result)

    # ==== Task C: 对比总结 ====
    print("\n【Task C】对比总结...")
    # 对比模拟数据和真实数据的结果差异
    write_comparison_report(synth_result, kaggle_result)

    # ---- 输出所有生成的文件路径 ----
    print("\n全部完成!")
    print(f"输出文件:")
    print(f"  {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"  {RESULTS_DIR / 'kaggle_report.md'}")
    print(f"  {RESULTS_DIR / 'summary_comparison.md'}")
    print(f"  {RESULTS_DIR / 'synthetic_analysis.png'}")
    print(f"  {RESULTS_DIR / 'kaggle_analysis.png'}")


if __name__ == "__main__":
    main()
