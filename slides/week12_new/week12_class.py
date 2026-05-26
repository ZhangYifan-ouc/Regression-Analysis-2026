# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["meta"]
# # 第12周：偏差-方差权衡与损失函数
#
# 今天不是来听课的——今天是来下注的。
#
# 你会在七幕里反复做同一件事：
# 先猜，再看，然后发现自己刚才猜错了，最后理解为什么。
#
# 课堂规则：
# - 先猜，再运行；
# - 先看图，再命名；
# - 先做判断，再听解释。

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note（全局）**
# - 每幕的 takeaway 不要在一开始就喊出来，要让学生先经历一遍“现象 → 命名 → 结论”。
# - 间奏 1 和间奏 2 是最适合让学生复述的关键节点，不要跳过。
# - 如果课时紧张，第 4 幕和第 6 幕可以压缩代码部分，但 takeaway 必须保留。

# %% tags=["stage"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20260523)


# %% tags=["stage"]
def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.2 * x) + 0.15 * x


def make_noisy_sample(n=80, noise_std=0.35, x_low=-3.0, x_high=3.0, rng=None):
    rng = RNG if rng is None else rng
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


def polynomial_model_mse(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def polynomial_model_mae(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", QuantileRegressor(quantile=0.5, solver="highs")),
        ]
    )


def fit_degree(x_train, y_train, x_eval, degree: int, loss: str = "mse"):
    model = (
        polynomial_model_mse(degree) if loss == "mse" else polynomial_model_mae(degree)
    )
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


# %% [markdown] tags=["cue"]
# ## 第 0 幕：为什么“训练误差更低”可以是陷阱？
#
# 假设你是某家连锁超市的数据团队负责人。
#
# 老板交给你一份历史销售数据，让你训练一个预测模型，用于明天开始的全国促销定价决策。
#
# 你拿了三个模型去汇报：
# - A 模型在训练数据上几乎零误差，每条记录都完美拟合；
# - B 模型在训练数据上误差中等；
# - C 模型在训练数据上误差最大。
#
# **你直觉上会觉得哪个最危险？**
#
# 今天这堂课会反复回到同一个判断：
# > 训练误差只是故事的一半，另一半是模型到了它没见过的世界之后，还能不能站稳。

# %% [markdown] tags=["cue"]
# ## 第 1 幕：三位候选人，谁该上线？
#
# 现在你有三个候选的预测模型：
# - `degree = 1`：只会画直线
# - `degree = 4`：有一定灵活性
# - `degree = 15`：能在训练点上扭来扭去
#
# 这三个模型都看了同一份训练数据。
#
# **先下注，不准翻后面的答案：**
# - 谁最像在抄答案而不是学习规律？
# - 谁最像根本没看懂题目？
# - 如果今天必须选一个上线，你敢选谁？

# %% tags=["stage"]
x, y = make_noisy_sample(n=120, noise_std=0.35, rng=np.random.default_rng(7))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.35, random_state=42
)

x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
y_true_grid = true_function(x_grid.ravel())

degrees_to_show = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, degree in zip(axes, degrees_to_show):
    y_grid_pred, model = fit_degree(x_train, y_train, x_grid, degree)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_rmse = root_mean_squared_error(y_train, train_pred)
    test_rmse = root_mean_squared_error(y_test, test_pred)

    ax.scatter(x_train[:, 0], y_train, s=18, alpha=0.6, label="train")
    ax.scatter(x_test[:, 0], y_test, s=18, alpha=0.6, label="test")
    ax.plot(
        x_grid[:, 0],
        y_true_grid,
        color="black",
        linewidth=2,
        linestyle="--",
        label="truth",
    )
    ax.plot(
        x_grid[:, 0],
        y_grid_pred,
        color="#d62728",
        linewidth=2.5,
        label=f"degree={degree}",
    )
    ax.set_title(
        f"degree={degree}\ntrain RMSE={train_rmse:.3f}, test RMSE={test_rmse:.3f}"
    )
    ax.set_xlabel("x")

axes[0].set_ylabel("y")
axes[-1].legend(loc="upper left", fontsize=10)
fig.suptitle("三位候选人：谁最像会在真实世界翻车？", y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["checkpoint"]
# **Checkpoint 1**
#
# 只根据刚才这三张图，口头回答：
#
# > 如果你今天必须选一个上线，你会选 `1 / 4 / 15` 中的哪一个？为什么？

# %% [markdown] tags=["cue"]
# ## 第 2 幕：完整成绩单
#
# 刚才只看了三个个案。现在把 1 到 18 阶全部拉出来。
#
# 从更加广阔的角度来趋势。
#
# **先猜：**
# - 训练误差会怎么走？
# - 测试误差会不会一路下降？
# - “最好的模型”会出现在左边、中间，还是很右边？

# %% tags=["stage"]
records = []
for degree in range(1, 19):
    model = polynomial_model_mse(degree)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    records.append(
        {
            "degree": degree,
            "train_rmse": root_mean_squared_error(y_train, y_train_pred),
            "test_rmse": root_mean_squared_error(y_test, y_test_pred),
        }
    )

error_df = pd.DataFrame(records)
error_df["generalization_gap"] = error_df["test_rmse"] - error_df["train_rmse"]

best_degree = int(error_df.loc[error_df["test_rmse"].idxmin(), "degree"])
worst_gap_degree = int(error_df.loc[error_df["generalization_gap"].idxmax(), "degree"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    error_df["degree"],
    error_df["train_rmse"],
    marker="o",
    linewidth=2.2,
    label="train RMSE",
)
ax.plot(
    error_df["degree"],
    error_df["test_rmse"],
    marker="o",
    linewidth=2.2,
    label="test RMSE",
)
ax.axvline(
    best_degree,
    color="gray",
    linestyle="--",
    alpha=0.75,
    label=f"best degree={best_degree}",
)
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("RMSE")
ax.set_title("完整成绩单：谁的测试表现最好？")
ax.legend()
plt.tight_layout()
plt.show()

scoreboard = (
    error_df.sort_values("test_rmse")
    .loc[:, ["degree", "train_rmse", "test_rmse", "generalization_gap"]]
    .head(8)
    .reset_index(drop=True)
)
scoreboard

print(f"Best test RMSE occurs at degree = {best_degree}")
print(f"Largest generalization gap occurs at degree = {worst_gap_degree}")

# %% [markdown] tags=["explain"]
# **解释**
#
# - 训练误差几乎一路下降。
# - 测试误差是一条 U 形曲线：先降后升。
# - 所以真正危险的不是“模型不够复杂”，而是“复杂到开始学噪声”。

# %% [markdown] tags=["takeaway"]
# **Takeaway：从图到判断**
#
# 这件事在统计里有一个正式名字：
#
# > 当模型复杂度过低时，它甚至无法正确捕捉真实规律——这叫 **high bias**（高偏差）。
# > 当模型复杂度过高时，它会过分跟随训练集的随机噪声——这叫 **high variance**（高方差）。
#
# 好模型不是追求某一项为零，而是让两者平衡。
#
# 所以，你刚才在成绩单里看到的那个 U 形，就是 **bias-variance tradeoff** 的直观形状。

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note**
# - 这是全课第一次正式命名 bias 和 variance，之前都压着不给术语。
# - 此时可以停 30 秒，让学生复述一下“**欠拟合**对应什么？**过拟合**对应什么？”
# - 如果学生答错，不要直接纠正，追问：你再看看 degree=1 和 degree=15 的图，它们各自失败的原因一样吗？

# %% [markdown] tags=["cue"]
# ## 第 3 幕：谁更抖？
#
# 真正麻烦的不是“模型复杂”，而是“模型不稳定”。
#
# 下面我们做一件事：**真实规律保持不变，但训练样本每次都不一样。**
#
# 就像同一个考试大纲，但每次给你不同的练习题——
# 一个好学生应该每次都能考出类似的分数，而不是一次满分一次不及格。
#
# **先猜：**
# - 低复杂度和高复杂度，谁的答案更容易因样本不同而剧烈波动？
# - 如果你是一位风险总监，你会更怕哪一类模型？

# %% tags=["stage"]
x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
y_eval_true = true_function(x_eval.ravel())

degree_predictions = {}
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, degree in zip(axes, [2, 15]):
    collected = []
    for sample_idx in range(14):
        x_sample, y_sample = make_noisy_sample(
            n=35, noise_std=0.35, rng=np.random.default_rng(1000 + sample_idx)
        )
        y_pred, _ = fit_degree(x_sample, y_sample, x_eval, degree)
        collected.append(y_pred)
        ax.plot(x_eval[:, 0], y_pred, alpha=0.30, linewidth=1.4)
    degree_predictions[degree] = np.vstack(collected)
    ax.plot(
        x_eval[:, 0],
        y_eval_true,
        color="black",
        linewidth=3,
        linestyle="--",
        label="truth",
    )
    ax.set_title(f"Repeated fits with degree={degree}")
    ax.set_xlabel("x")
    ax.legend(loc="upper left")

axes[0].set_ylabel("predicted y")
fig.suptitle("同一真实规律、不同训练样本：谁在剧烈抖动？", y=1.05)
plt.tight_layout()
plt.show()

variance_summary = []
for degree, preds in degree_predictions.items():
    pointwise_std = preds.std(axis=0)
    variance_summary.append(
        {
            "degree": degree,
            "mean_prediction_std": pointwise_std.mean(),
            "max_prediction_std": pointwise_std.max(),
        }
    )

variance_df = pd.DataFrame(variance_summary)
variance_df

# %% [markdown] tags=["explain"]
# **解释**
#
# - 低复杂度模型的多次拟合结果之间更接近。
# - 高复杂度模型的曲线抖动剧烈。
# - 这意味着：样本一变，它的预测就变——在业务上就是不可靠。

# %% [markdown] tags=["takeaway"]
# **Takeaway：variance 不是公式，而是风险**
#
# 高方差模型的危险不是“它在训练集上不好”，而是：
#
# > **它对训练样本过于敏感，导致它在生产环境里不可预测。**
#
# 在生产系统中，我们宁可接受一个“有偏但稳定”的模型，也不愿意接受“每次上线都不同”的模型。
# 这就是为什么 bias-variance tradeoff 不是一个纯理论问题——它直接影响线上决策安全。

# %% [markdown] tags=["checkpoint"]
# **Checkpoint**
#
# 请补全这句话：
#
# > 高 variance model 的危险，不是它不会拟合训练集，而是它对 ______ 过于敏感。

# %% [markdown] tags=["interlude"]
# ## —— 间奏 1：到目前为止的三件事 ——
#
# 停下来收回一下。
#
# 我们已经看到了三个彼此关联的判断：
#
# 1. **训练误差下降不代表泛化能力更好。**
# 2. **模型太简单会欠拟合（high bias），太复杂会过拟合（high variance）。**
# 3. **高方差模型的输出不稳定，在业务上意味着不可靠。**
#
# 这三个判断指向同一个核心结论：
#
# > **模型选择不是选拟合更狠的，而是选在新数据上更稳的。**
#
# 下半节课，我们会换一个问题：
# > 用什么标准来定义“多好才算好”？

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note**
# - 间奏 1 是一个关键的“认知锚”，不要跳过。
# - 可以让学生自己复述：前三幕分别学到了什么？
# - 如果学生总结不完整，补一句：为什么 bias-variance tradeoff 不是公式，而是选择？
# - 接下来进入损失函数部分，语义转向“怎样才算好”，节奏可以稍微放缓。

# %% [markdown] tags=["cue"]
# ## 第 4 幕：换一把尺子
#
# 我们一直在用 RMSE 来评判模型。
#
# 但 RMSE 并不是唯一的选择。
#
# RMSE 和 MAE 之间，不是“谁算得更准”的区别，而是：
#
# > **你对错误的态度不一样。**
#
# 先看正常数据下的对比。
#
# **先猜：**
# - 如果所有误差都差不多大小，RMSE 和 MAE 会给出类似的结论吗？
# - 它们各自更关注什么样的错误？

# %% tags=["stage"]
y_true = np.array([10, 11, 9, 10, 10, 11, 10, 9], dtype=float)
y_pred_a = np.array([10, 11, 9, 10, 10, 11, 10, 9], dtype=float)
y_pred_b = np.array([10, 10, 11, 10, 10, 10, 10, 10], dtype=float)
y_pred_c = np.array([11, 10, 9, 11, 9, 10, 11, 9], dtype=float)

scenarios_clean = pd.DataFrame(
    {
        "scenario": ["perfect", "one small bias", "small scatter"],
        "RMSE": [
            root_mean_squared_error(y_true, y_pred_a),
            root_mean_squared_error(y_true, y_pred_b),
            root_mean_squared_error(y_true, y_pred_c),
        ],
        "MAE": [
            mean_absolute_error(y_true, y_pred_a),
            mean_absolute_error(y_true, y_pred_b),
            mean_absolute_error(y_true, y_pred_c),
        ],
    }
)

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(scenarios_clean))
width = 0.35
ax.bar(x_pos - width / 2, scenarios_clean["RMSE"], width=width, label="RMSE")
ax.bar(x_pos + width / 2, scenarios_clean["MAE"], width=width, label="MAE")
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios_clean["scenario"])
ax.set_ylabel("Error")
ax.set_title("Normal data: RMSE and MAE are often close")
ax.legend()
plt.tight_layout()
plt.show()

scenarios_clean

# %% [markdown] tags=["explain"]
# **解释**
#
# - 当误差均匀分布时，RMSE 和 MAE 给出的相对大小往往接近。
# - 但它们在数学上优化的并不是同一个东西：
#   - 最小化 MSE 等价于拟合条件期望；
#   - 最小化 MAE 等价于拟合条件中位数。

# %% [markdown] tags=["takeaway"]
# **Takeaway**
#
# RMSE 和 MAE 的区别不是一个公式的差别——它们代表了不同的建模目标。
#
# 即使在无异常值的数据上，选择 RMSE 还是 MAE，已经在回答：
# > 你想让模型更贴近“平均值”，还是更贴近“最典型的点”？

# %% [markdown] tags=["cue"]
# ## 第 5 幕：一个离群点，先刺穿谁？
#
# 现在我们把一个很大的错误放进预测里。
#
# 不改变数据，只改变其中一个预测值。
#
# **先下注：**
# - RMSE 和 MAE，谁先爆表？
# - 如果你做的是风控、预算预测或医疗告警，哪个指标更符合你对风险的心理感受？

# %% tags=["stage"]
y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
y_pred_outlier = y_pred_clean.copy()
y_pred_outlier[-1] = 80

metrics_df = pd.DataFrame(
    {
        "scenario": ["clean prediction", "one large outlier"],
        "RMSE": [
            root_mean_squared_error(y_true, y_pred_clean),
            root_mean_squared_error(y_true, y_pred_outlier),
        ],
        "MAE": [
            mean_absolute_error(y_true, y_pred_clean),
            mean_absolute_error(y_true, y_pred_outlier),
        ],
    }
)
metrics_df["RMSE/MAE ratio"] = metrics_df["RMSE"] / metrics_df["MAE"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].scatter(range(len(y_true)), y_true, s=85, label="true")
axes[0].scatter(range(len(y_true)), y_pred_outlier, s=85, label="pred")
axes[0].set_title("One outlier changes one prediction")
axes[0].set_xlabel("sample index")
axes[0].set_ylabel("value")
axes[0].legend()

x_pos = np.arange(len(metrics_df))
width = 0.35
axes[1].bar(x_pos - width / 2, metrics_df["RMSE"], width=width, label="RMSE")
axes[1].bar(x_pos + width / 2, metrics_df["MAE"], width=width, label="MAE")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(metrics_df["scenario"], rotation=10)
axes[1].set_title("Which metric gets hit harder?")
axes[1].set_ylabel("metric value")
axes[1].legend()
plt.tight_layout()
plt.show()

metrics_df

# %% [markdown] tags=["explain"]
# **解释**
#
# - RMSE 因为平方惩罚，会被单个大错快速拉高。
# - MAE 也上升，但不会被极端错误那样强烈地主导。
# - 所以 RMSE 更像一个“不能容忍大错”的代价函数；MAE 更像一个“更关心典型表现”的代价函数。

# %% [markdown] tags=["takeaway"]
# **Takeaway：选择指标就是选择你对大错的态度**
#
# > RMSE 惩罚大错远远超过 MAE。
#
# 这不是数学问题，这是风险偏好问题。
#
# 如果你在风控 / 医疗 / 预算等场景中，**一次大错的代价远超多次小错之和**，你会自然更关心 RMSE。
# 如果你的数据本身就经常有异常值，而你对异常值没有那么强的惩罚需求，那么 MAE 更自然。
#
# 指标选择，在业务层就是在定义什么叫“不能接受”。

# %% [markdown] tags=["checkpoint"]
# **Checkpoint**
#
# 假设你在一家外卖平台做定价策略：
# - 大部分时间的预测误差都很小；
# - 但偶尔一次大错，会导致整个城市的配送资源被错误调度。
#
# > 这种情况下，你会更关心 RMSE 还是 MAE？为什么？

# %% [markdown] tags=["cue"]
# ## 第 6 幕：损失函数不只用来打分，它也决定模型长什么样
#
# 前两幕我们只把 RMSE 和 MAE 当评估指标。
#
# 但真正重要的问题是：
#
# > **如果我在训练阶段就直接最小化 MAE，而不是最小化 MSE，训练出来的模型会不一样吗？**
#
# 也就是说——损失函数不只是评分表，它也是建模的决策依据。
#
# 下面我们在同一组数据上分别用 MSE 和 MAE 训练多项式模型，看看会发生什么。

# %% tags=["stage"]
x_sim, y_sim = make_noisy_sample(n=100, noise_std=0.30, rng=np.random.default_rng(42))
y_sim_contaminated = y_sim.copy()
y_sim_contaminated[85] += 4.5
y_sim_contaminated[90] -= 4.0

x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
    x_sim, y_sim_contaminated, test_size=0.35, random_state=123
)

degree_compare = 6
y_grid_mse, _ = fit_degree(x_train_c, y_train_c, x_grid, degree_compare, loss="mse")
y_grid_mae, _ = fit_degree(x_train_c, y_train_c, x_grid, degree_compare, loss="mae")

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    x_train_c[:, 0], y_train_c, s=24, alpha=0.7, label="train (with outliers)", zorder=5
)
ax.scatter(x_test_c[:, 0], y_test_c, s=24, alpha=0.7, label="test", zorder=4)
ax.plot(
    x_grid[:, 0],
    y_true_grid,
    color="black",
    linewidth=2,
    linestyle="--",
    label="true function",
)
ax.plot(x_grid[:, 0], y_grid_mse, color="#d62728", linewidth=2.8, label="MSE fit")
ax.plot(x_grid[:, 0], y_grid_mae, color="#1f77b4", linewidth=2.8, label="MAE fit")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Same data, same degree={degree_compare}: loss function shapes the model")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

mse_train_rmse = root_mean_squared_error(
    y_train_c,
    polynomial_model_mse(degree_compare).fit(x_train_c, y_train_c).predict(x_train_c),
)
mse_test_rmse = root_mean_squared_error(
    y_test_c,
    polynomial_model_mse(degree_compare).fit(x_train_c, y_train_c).predict(x_test_c),
)
mae_train_rmse = root_mean_squared_error(
    y_train_c,
    polynomial_model_mae(degree_compare).fit(x_train_c, y_train_c).predict(x_train_c),
)
mae_test_rmse = root_mean_squared_error(
    y_test_c,
    polynomial_model_mae(degree_compare).fit(x_train_c, y_train_c).predict(x_test_c),
)

comparison_df = pd.DataFrame(
    {
        "loss": ["MSE (standard)", "MAE (quantile median)"],
        "train_RMSE": [mse_train_rmse, mae_train_rmse],
        "test_RMSE": [mse_test_rmse, mae_test_rmse],
    }
)
comparison_df

# %% tags=["stage"]
mse_records = []
mae_records = []
for degree in range(1, 19):
    model_mse = polynomial_model_mse(degree)
    model_mae = polynomial_model_mae(degree)
    model_mse.fit(x_train_c, y_train_c)
    model_mae.fit(x_train_c, y_train_c)
    mse_records.append(
        {
            "degree": degree,
            "test_rmse": root_mean_squared_error(y_test_c, model_mse.predict(x_test_c)),
            "loss": "MSE",
        }
    )
    mae_records.append(
        {
            "degree": degree,
            "test_rmse": root_mean_squared_error(y_test_c, model_mae.predict(x_test_c)),
            "loss": "MAE",
        }
    )

sweep_df = pd.DataFrame(mse_records + mae_records)
mse_subset = sweep_df[sweep_df["loss"] == "MSE"]
mae_subset = sweep_df[sweep_df["loss"] == "MAE"]
best_mse_degree = int(mse_subset.loc[mse_subset["test_rmse"].idxmin(), "degree"])
best_mae_degree = int(mae_subset.loc[mae_subset["test_rmse"].idxmin(), "degree"])

fig, ax = plt.subplots(figsize=(10, 6))
for loss_label, marker, color in [("MSE", "o", "#d62728"), ("MAE", "s", "#1f77b4")]:
    subset = sweep_df[sweep_df["loss"] == loss_label]
    ax.plot(
        subset["degree"],
        subset["test_rmse"],
        marker=marker,
        linewidth=2.2,
        color=color,
        label=f"{loss_label}",
    )

ax.axvline(
    best_mse_degree,
    color="#d62728",
    linestyle="--",
    alpha=0.7,
    label=f"best MSE degree={best_mse_degree}",
)
ax.axvline(
    best_mae_degree,
    color="#1f77b4",
    linestyle="--",
    alpha=0.7,
    label=f"best MAE degree={best_mae_degree}",
)
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("Test RMSE")
ax.set_title("Choosing best complexity: MSE vs MAE may disagree")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Best degree by MSE = {best_mse_degree}, by MAE = {best_mae_degree}")

# %% [markdown] tags=["explain"]
# **解释**
#
# - 在存在离群点的数据上，MSE 最优的拟合曲线被离群点明显拉偏；
# - MAE 最优的拟合曲线更抗拒离群点的拖拽，更贴近真实规律；
# - 并且，MSE 和 MAE 可能选出不同的“最优复杂度”。

# %% [markdown] tags=["takeaway"]
# **Takeaway：损失函数决定了你最终得到什么模型**
#
# 损失函数不只是评估标准——它从根本上参与了建模。
#
# 即使你所有的数据和特征都一样，当你从 MSE 切换到 MAE：
# - 训练出来的系数可能不同；
# - 选出来的最优复杂度可能不同；
# - 模型对异常值的响应完全不同。
#
# 所以选择损失函数，本质上是在选择：
#
# > **“你希望模型去逼近什么？”**

# %% [markdown] tags=["interlude"]
# ## —— 间奏 2：今天的七个判断 ——
#
# 收回今天全部内容。
#
# 1. 训练误差下降，不代表泛化能力更好。
# 2. 模型太简单 → high bias；模型太复杂 → high variance。
# 3. 高方差在生产环境中意味着不可预测和不可靠。
# 4. 即使在正常数据上，RMSE 和 MAE 也代表了不同的优化目标。
# 5. 异常值会让 RMSE 快速膨胀，选择 RMSE 或 MAE 就是选择对大错的态度。
# 6. 损失函数不只是评分表——它会改变模型本身的样子和最优复杂度。
#
# 这些判断汇聚成一句话：
#
# > **建模的核心不是拟合，而是选择——选择复杂度、选择损失、选择你愿意承担的风险。**

# %% [markdown] tags=["meta"]
# ## 下一讲自然问题
#
# 今天我们看到的最核心矛盾是：
#
# > 模型越复杂，越可能 high variance，越不稳定。
#
# 那下一步自然就会问：
#
# > **能不能在建模的时候就主动给模型“加约束”，强迫它不要太复杂？**
#
# 这正是 Ridge / Lasso / Elastic Net 的起点。
#
# 第 13 周，我们会把所有 loss 和 penalty 放进一个统一的框架：
#
# \[
# \min_\beta \ \ell(y, X\beta) + \lambda \cdot P(\beta)
# \]
#
# 那时你会发现，今天看到的所有现象——过拟合、不稳定、损失选择——都可以被这一个式子重新理解。

# %% [markdown] tags=["backup"]
# ## Backup
#
# 可选延伸：
# - 改变训练样本量，观察方差曲线的变化；
# - 改变噪声强度，观察 U 形曲线的移动；
# - 多次加入不同数量的离群点，观察 MSE 和 MAE 最优度数的分歧是否扩大；
# - 使用 Huber Loss 作为 MSE 和 MAE 之间的折中，观察模型行为。
