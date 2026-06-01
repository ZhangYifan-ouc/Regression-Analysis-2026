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
# # Week 13：正则化回归与变量筛选
#
# **主题**：`Ridge`、`Lasso`、`Elastic Net`
#
# **本节课主线**：
# - 为什么 OLS 在某些场景下不够稳？
# - `loss + penalty` 如何改变模型性格？
# - 为什么 `Ridge`、`Lasso`、`Elastic Net` 会走出不同的系数路径？
# - 为什么模型选择不能只看图，还要看交叉验证结果？
#
# **使用说明**：
# - 本 notebook 以 `week13_class.py` 为 source of truth；
# - 课堂结构按“看图提问 → 运行代码 → 总结判断”组织；
# - 当前版本先搭建课堂骨架，后续再继续细化数据、图像与脚本。

# %% [markdown]
# ## Prologue：为什么我们还要继续折腾线性模型？
#
# 上节课我们讲了 Bias-Variance Trade-off。
# 如果借用“开车”的比喻：
#
# - Bias 高，像车开不快；
# - Variance 高，像车开不稳；
# - 真正想要的是：又快又稳；
# - 最终还得去没见过的赛道上看表现，也就是泛化能力。
#
# 这节课我们继续以线性模型为例，不是因为它最强，而是因为它足够透明。
# 正因为透明，我们才能看见：
#
# - OLS 在某些场景下为什么会不稳定；
# - penalty 如何改变模型；
# - 为什么不同正则化方法会给出不同的变量行为。

# %% [markdown] tags=["script", "teacher-only"]
# 这一节的节奏建议：
# - 第一幕先立住“OLS 不是错，而是不够稳”；
# - 不要急着讲公式推导，先让“不稳定”这件事被看见；
# - 之后每一幕都尽量围绕一张主图展开。

# %% [markdown]
# ---
# ## 第一幕：OLS 还不够——为什么模型需要“约束”？
#
# 这一幕只做一件事：
# **先接受一个判断：在相关特征存在时，OLS 的结论可能不够稳。**

# %% [markdown] tags=["prompt"]
# 如果你要把一个回归模型交给业务团队长期使用，你更担心哪一件事？
#
# 1. 训练误差还不够低；
# 2. 换一批样本后，模型系数和结论明显变化。
#
# 先不要运行代码，先凭直觉选一个。

# %% [markdown] tags=["stage"]
# **本幕主图**：OLS 系数不稳定图
#
# 这张图要回答：
# > 同一个建模任务里，只要样本略有变化，OLS 的系数会不会明显波动？
#
# 这里用“多次切分下的 OLS 系数波动”来把“不稳定”直接画出来。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False


RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

FEATURE_NAMES = [
    "x1_signal",
    "x2_collinear",
    "x3_collinear",
    "x4_noise",
    "x5_noise",
    "x6_noise",
]
TRUE_COEF = np.array([3.0, 0.0, 0.0, 0.8, 0.0, 0.0])


def make_correlated_regression_data(n_samples=180, noise_std=1.0, seed=RANDOM_SEED):
    rng_local = np.random.default_rng(seed)
    latent = rng_local.normal(size=n_samples)

    x1 = latent + rng_local.normal(scale=0.18, size=n_samples)
    x2 = latent + rng_local.normal(scale=0.18, size=n_samples)
    x3 = 0.8 * latent + rng_local.normal(scale=0.22, size=n_samples)
    x4 = rng_local.normal(size=n_samples)
    x5 = rng_local.normal(size=n_samples)
    x6 = rng_local.normal(size=n_samples)

    X = np.column_stack([x1, x2, x3, x4, x5, x6])
    y = X @ TRUE_COEF + rng_local.normal(scale=noise_std, size=n_samples)
    return X, y


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


X, y = make_correlated_regression_data()

n_repeats = 80
coef_records = []

for repeat in range(n_repeats):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=repeat)
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    for feature_name, coef_value in zip(FEATURE_NAMES, ols.coef_):
        coef_records.append({"feature": feature_name, "coefficient": coef_value})

coef_df = pd.DataFrame(coef_records)
feature_order = FEATURE_NAMES
positions = np.arange(len(feature_order))

fig, ax = plt.subplots(figsize=(11, 6))
for idx, feature in enumerate(feature_order):
    feature_values = coef_df.loc[
        coef_df["feature"] == feature, "coefficient"
    ].to_numpy()
    jitter = np.linspace(-0.18, 0.18, len(feature_values))
    ax.scatter(
        np.full_like(feature_values, idx, dtype=float) + jitter,
        feature_values,
        alpha=0.35,
        s=24,
    )
    ax.boxplot(
        feature_values,
        positions=[idx],
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor="#dbeafe", alpha=0.75),
        medianprops=dict(color="#1d4ed8", linewidth=2),
        whiskerprops=dict(color="#1d4ed8"),
        capprops=dict(color="#1d4ed8"),
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(positions)
ax.set_xticklabels(feature_order, rotation=20)
ax.set_ylabel("OLS coefficient across repeated splits")
ax.set_title("OLS 在相关特征场景下会出现明显的系数波动")
ax.set_xlabel("feature")
plt.tight_layout()
plt.show()

# %% [markdown] tags=["takeaway"]
# OLS 的问题不是“完全不能用”，而是：
#
# - 当特征相关、样本扰动存在时，系数可能明显波动；
# - 这意味着训练集上拟合得不错，并不代表模型结论足够稳定；
# - 所以今天我们要讨论的，不是“换一个完全不同的模型”，而是“怎样给模型加约束”。

# %% [markdown] tags=["script", "teacher-only"]
# 这里先把 stakes 立住，不急着进入术语。
# 如果现场对“不稳定”这件事还不够敏感，可以追问：
# “如果这周说价格重要，下周说广告重要，这个模型你敢交给老板吗？”

# %% [markdown]
# ---
# ## 第二幕：统一框架——`loss + penalty`
#
# 这一幕要建立一个统一语言：
# **正则化不是另起炉灶，而是在原来的拟合目标里，显式加入复杂度成本。**

# %% [markdown] tags=["prompt"]
# 如果我们已经能最小化误差，为什么还要故意给模型“加限制”？
#
# 你愿不愿意为了更稳，牺牲一点训练集上的贴合度？

# %% [markdown] tags=["stage"]
# **本幕主图**：`loss + penalty` 的结构图
#
# 这张图要回答：
# > penalty 到底在干什么？
#
# 这里用一张板书式结构图，把 `loss` 和 `penalty` 的职责拆开。

# %%
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

loss_box = patches.FancyBboxPatch(
    (0.8, 1.35),
    2.5,
    1.2,
    boxstyle="round,pad=0.2",
    facecolor="#dbeafe",
    edgecolor="#2563eb",
    linewidth=2,
)
penalty_box = patches.FancyBboxPatch(
    (4.0, 1.35),
    2.5,
    1.2,
    boxstyle="round,pad=0.2",
    facecolor="#fee2e2",
    edgecolor="#dc2626",
    linewidth=2,
)
total_box = patches.FancyBboxPatch(
    (7.2, 1.1),
    2.0,
    1.7,
    boxstyle="round,pad=0.25",
    facecolor="#ecfccb",
    edgecolor="#65a30d",
    linewidth=2.2,
)

for patch in [loss_box, penalty_box, total_box]:
    ax.add_patch(patch)

ax.text(
    2.05,
    1.95,
    "loss",
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    color="#1d4ed8",
)
ax.text(2.05, 1.45, "fit the data", ha="center", va="center", fontsize=12)
ax.text(
    5.25,
    1.95,
    "penalty",
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    color="#b91c1c",
)
ax.text(5.25, 1.45, "control coefficient size", ha="center", va="center", fontsize=12)
ax.text(
    8.2,
    2.0,
    "objective",
    ha="center",
    va="center",
    fontsize=17,
    weight="bold",
    color="#3f6212",
)
ax.text(8.2, 1.5, "loss + penalty", ha="center", va="center", fontsize=13)

ax.text(3.55, 1.95, "+", fontsize=24, weight="bold", ha="center", va="center")
ax.annotate(
    "same model family",
    xy=(2.05, 2.85),
    xytext=(5.25, 3.35),
    arrowprops=dict(arrowstyle="->", lw=1.8),
)
ax.annotate(
    "pay some fit to gain control",
    xy=(5.3, 1.0),
    xytext=(7.55, 0.4),
    arrowprops=dict(arrowstyle="->", lw=1.8),
)
ax.set_title("正则化不是另起炉灶，而是在 loss 外显式加入复杂度成本", pad=14)

plt.tight_layout()
plt.show()

# %% [markdown] tags=["takeaway"]
# 正则化回归的共同语言是：
#
# `objective = loss + penalty`
#
# 这意味着：
# - `loss` 负责拟合数据；
# - `penalty` 负责约束复杂度；
# - 后面三种方法的真正差异，不在于“是不是回归”，而在于“怎样约束系数”。

# %% [markdown] tags=["script", "teacher-only"]
# 这一幕不要展开太多优化细节。
# 任务只有一个：把“正则化 = 显式付出复杂度成本”这个判断立住。

# %% [markdown]
# ---
# ## 第三幕：Ridge vs Lasso——同样收缩，为什么一个保留全部、一个开始筛选？
#
# 这是全节课最重要的视觉对比幕。
# 主图负责展示系数路径差异，辅图负责把路径变化连接回 bias-variance / 稳定性主线。

# %% [markdown] tags=["prompt"]
# 看图之前先猜：
#
# 如果我们不断增大 penalty，下面两种现象你觉得哪一个更可能出现？
#
# 1. 所有变量一起慢慢缩小；
# 2. 有些变量会被直接“挤出场”。
#
# 你觉得这两种行为，对业务上的“变量名单”意味着什么不同？

# %% [markdown] tags=["stage"]
# **本幕主图**：`Ridge` 与 `Lasso` 的 regularization path 并排对比图
#
# 这张图要回答：
# > 同样都是 penalty，为什么一个更像“整体收缩”，另一个更像“变量筛选”？
#
# 这里直接并排展示：
# - `Ridge` regularization path；
# - `Lasso` regularization path；
# - 并保留一组强相关特征，为后续 `Elastic Net` 铺垫。

# %%
X_path, y_path = make_correlated_regression_data(
    n_samples=220, noise_std=1.0, seed=RANDOM_SEED
)
X_train_path, X_test_path, y_train_path, y_test_path = train_test_split(
    X_path, y_path, test_size=0.3, random_state=RANDOM_SEED
)

alphas = np.logspace(-4, 3, 80)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ]
    )
    ridge_model.fit(X_train_path, y_train_path)
    ridge_coefs.append(ridge_model.named_steps["model"].coef_)

    lasso_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_SEED)),
        ]
    )
    lasso_model.fit(X_train_path, y_train_path)
    lasso_coefs.append(lasso_model.named_steps["model"].coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
log_alphas_desc = np.log10(alphas[::-1])
for i, feature_name in enumerate(FEATURE_NAMES):
    axes[0].plot(log_alphas_desc, ridge_coefs[::-1, i], label=feature_name)
    axes[1].plot(log_alphas_desc, lasso_coefs[::-1, i], label=feature_name)

axes[0].set_title("Ridge：penalty 越强，系数越一起向 0 收缩")
axes[1].set_title("Lasso：penalty 越强，部分变量会被压到 0")
for ax in axes:
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(alpha)  ← penalty stronger")
    ax.set_ylabel("standardized coefficient")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(
    "随着 alpha 增大，Ridge 与 Lasso 会展现出不同的收缩方式", y=1.02, fontsize=14
)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["stage"]
# **本幕辅图**：Bias-Variance 权衡中的动图或稳定性辅助图
#
# 这张辅图要回答：
# > 当 penalty 逐渐变大时，我们到底是怎样换来更稳定模型的？
#
# 这里用“系数波动 + 误差变化”的辅助图，把路径变化重新连接回“variance 下降、模型更稳”的主线。

# %%
stability_alphas = np.logspace(-4, 3, 16)
repeated_splits = range(45)
records = []

for alpha in stability_alphas:
    train_rmse_values = []
    test_rmse_values = []
    coef_matrix = []
    for split_seed in repeated_splits:
        X_train, X_test, y_train, y_test = train_test_split(
            X_path, y_path, test_size=0.3, random_state=split_seed
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
            ]
        )
        model.fit(X_train, y_train)
        train_rmse_values.append(rmse(y_train, model.predict(X_train)))
        test_rmse_values.append(rmse(y_test, model.predict(X_test)))
        coef_matrix.append(model.named_steps["model"].coef_)

    coef_matrix = np.array(coef_matrix)
    records.append(
        {
            "alpha": alpha,
            "train_rmse_mean": np.mean(train_rmse_values),
            "test_rmse_mean": np.mean(test_rmse_values),
            "coef_instability": coef_matrix[:, 0].std(),
        }
    )

stability_df = pd.DataFrame(records)

fig, ax1 = plt.subplots(figsize=(10.5, 6))
ax2 = ax1.twinx()

x_values = np.log10(stability_df["alpha"])
ax1.plot(
    x_values,
    stability_df["train_rmse_mean"],
    color="#2563eb",
    marker="o",
    label="mean train RMSE",
)
ax1.plot(
    x_values,
    stability_df["test_rmse_mean"],
    color="#dc2626",
    marker="o",
    label="mean test RMSE",
)
ax2.plot(
    x_values,
    stability_df["coef_instability"],
    color="#16a34a",
    marker="s",
    linestyle="--",
    label="std of x1 coefficient across splits",
)

ax1.set_xlabel("log10(alpha)")
ax1.set_ylabel("average RMSE")
ax2.set_ylabel("coefficient instability")
ax1.set_title("penalty 变大时：训练贴合度上升一些，但核心系数在不同切分下更稳定")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")
plt.tight_layout()
plt.show()

# %% [markdown] tags=["takeaway"]
# 这一幕至少要收住三个判断：
#
# - `Ridge` 的强项是稳定收缩，不是变量筛选；
# - `Lasso` 的强项是稀疏与筛选，但这也意味着变量名单会变化；
# - penalty 不是白加的：它的代价是牺牲一部分贴合度，换来更保守、更稳定的模型行为。

# %% [markdown] tags=["script", "teacher-only"]
# 这一幕不要陷入公式细节。
# 重点是先从图上读出“模型性格”。
# 如果时间允许，再追问：
# “业务上你要的是更稳，还是更短的变量名单？”

# %% [markdown]
# ## Synthesis 1
#
# 到这里，前半节课可以被压缩成一句话：
#
# > 正则化不是在背三个新名词，而是在同一个优化框架下，用不同 penalty 交换不同的模型性格。

# %% [markdown]
# ---
# ## 第四幕：Elastic Net——当变量高度相关时，折中为什么更成熟？
#
# 前一幕已经看到：Ridge 更稳，Lasso 更稀疏。
# 但如果变量彼此相关、成组出现，问题就变成：
# **只留一个代表，真的总是合理吗？**

# %% [markdown] tags=["prompt"]
# 如果两列变量其实表达的是相近信息，模型只留下其中一个，你会觉得这个结果：
#
# 1. 很干脆，很好；
# 2. 可能有点武断。
#
# 先选一个立场，再看图。

# %% [markdown] tags=["stage"]
# **本幕主图**：`Lasso` 与 `Elastic Net` 在相关变量场景下的路径对比图
#
# 这张图要回答：
# > 当变量成组相关时，模型是在“选一个代表”，还是“让一组变量共同保留并收缩”？
#
# 这里直接比较 `Lasso` 与 `Elastic Net` 的路径，并高亮相关变量组的行为差异。

# %%
from sklearn.linear_model import ElasticNet

elastic_alphas = np.logspace(-3, 1.0, 55)
lasso_group_coefs = []
elastic_group_coefs = []
elastic_l1_ratio = 0.45

for alpha in elastic_alphas:
    lasso_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_SEED)),
        ]
    )
    lasso_model.fit(X_train_path, y_train_path)
    lasso_group_coefs.append(lasso_model.named_steps["model"].coef_)

    elastic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=elastic_l1_ratio,
                    max_iter=20000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    elastic_model.fit(X_train_path, y_train_path)
    elastic_group_coefs.append(elastic_model.named_steps["model"].coef_)

lasso_group_coefs = np.array(lasso_group_coefs)
elastic_group_coefs = np.array(elastic_group_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for i, feature_name in enumerate(FEATURE_NAMES):
    line_alpha = 1.0 if i < 3 else 0.45
    axes[0].plot(
        np.log10(elastic_alphas),
        lasso_group_coefs[:, i],
        label=feature_name,
        alpha=line_alpha,
    )
    axes[1].plot(
        np.log10(elastic_alphas),
        elastic_group_coefs[:, i],
        label=feature_name,
        alpha=line_alpha,
    )

axes[0].set_title("Lasso：相关变量里更容易只留一个代表")
axes[1].set_title("Elastic Net：相关变量更可能共同保留并收缩")
for ax in axes:
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("standardized coefficient")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(
    "相关变量成组出现时，Elastic Net 往往比纯 Lasso 更稳健", y=1.02, fontsize=14
)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["stage"]
# **本幕辅图**：Elastic Net 参数角色的辅助说明图或小表
#
# 这张辅图只做说明：
# - 一个参数控制总体惩罚强度；
# - 一个参数控制更偏向 `Lasso` 还是更偏向 `Ridge`。

# %%
param_summary = pd.DataFrame(
    {
        "parameter": ["alpha", "l1_ratio"],
        "role": [
            "控制总体惩罚强度：越大，模型越保守",
            "控制更像 Lasso 还是更像 Ridge",
        ],
        "classroom_reading": [
            "刹车踩多大",
            "更想要筛选，还是更想要稳定",
        ],
    }
)

param_summary

# %% [markdown] tags=["takeaway"]
# `Elastic Net` 的价值不在于“又多了一个名词”，而在于：
#
# - 当变量成组相关时，纯 `Lasso` 可能显得过于武断；
# - `Elastic Net` 提供了“既想筛选，又不想过度丢掉相关信息”的折中；
# - 所以方法选择不能只看谁更稀疏，还要看数据结构是否支持这种稀疏。

# %% [markdown] tags=["script", "teacher-only"]
# 这一幕的重点是“相关变量成组”的现实感。
# 不要把时间花在复杂公式上，而是要把一个现实感立住：
# 现实问题里，很多变量不是单打独斗出现的。

# %% [markdown]
# ---
# ## 第五幕：会用比会看更重要——`GridSearchCV` 如何帮我们选 `alpha`？
#
# 前面几幕让我们看到不同 penalty 会带来不同模型行为。
# 这一幕要回答的是：
# **那到底该选多大的 penalty？**

# %% [markdown] tags=["prompt"]
# 如果一个模型更稀疏、变量更少，我们就应该优先选它吗？
#
# 还是说，真正该看的不是“看起来够简洁”，而是“在验证里表现如何”？

# %% [markdown] tags=["stage"]
# **本幕主图**：`GridSearchCV` / CV 分数随 `alpha` 变化的主图
#
# 这张图要回答：
# > `alpha` 应该怎么选？是越大越好吗，还是越稀疏越好吗？
#
# 这里直接展示 `GridSearchCV` 搜索结果，让最低点而不是“最强惩罚”成为视觉焦点。

# %%
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
search_alphas = np.logspace(-4, 3, 60)

search_spaces = {
    "Ridge": {
        "pipeline": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_SEED))]
        ),
        "param_grid": {"model__alpha": search_alphas},
    },
    "Lasso": {
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=20000, random_state=RANDOM_SEED)),
            ]
        ),
        "param_grid": {"model__alpha": search_alphas},
    },
    "Elastic Net": {
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        l1_ratio=0.45,
                        max_iter=20000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "param_grid": {"model__alpha": search_alphas},
    },
}

search_results = {}
for model_name, config in search_spaces.items():
    search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=None,
        return_train_score=True,
    )
    search.fit(X_path, y_path)
    search_results[model_name] = search

fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
colors = {"Ridge": "#2563eb", "Lasso": "#dc2626", "Elastic Net": "#16a34a"}

for ax, model_name in zip(axes, ["Ridge", "Lasso", "Elastic Net"]):
    search = search_results[model_name]
    result_df = pd.DataFrame(search.cv_results_)
    x_vals = result_df["param_model__alpha"].astype(float)
    y_vals = -result_df["mean_test_score"]
    x_plot = np.arange(len(x_vals))

    ax.plot(x_plot, y_vals, marker="o", color=colors[model_name])
    best_alpha = float(search.best_params_["model__alpha"])
    best_idx = int(np.argmin(np.abs(x_vals - best_alpha)))
    best_score = -search.best_score_
    ax.scatter(
        best_idx,
        best_score,
        color=colors[model_name],
        s=90,
        edgecolor="black",
        zorder=5,
    )
    ax.axvline(best_idx, color=colors[model_name], linestyle="--", alpha=0.7)
    y_margin = max(0.01, (y_vals.max() - y_vals.min()) * 0.25)
    ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)
    tick_idx = np.linspace(0, len(x_vals) - 1, 6, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{x_vals[i]:.4f}" for i in tick_idx], rotation=20)
    ax.set_ylabel("CV RMSE")
    ax.set_title(f"{model_name}：验证误差最低点出现在特定 alpha，而不是越大越好")

axes[-1].set_xlabel("alpha candidates in GridSearchCV")
fig.suptitle(
    "GridSearchCV 的任务是找到验证误差最低点，而不是追求最强惩罚", y=0.995, fontsize=14
)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["stage"]
# ###### **本幕辅图（可选）**：最终模型对比结果表
#
# 如果保留，这张辅图只做总结：
# - 哪个模型更偏向稳定预测；
# - 哪个模型更偏向变量筛选；
# - 哪个模型在相关变量场景下更稳健。

# %%
summary_rows = []
for model_name, search in search_results.items():
    best_model = search.best_estimator_
    best_alpha = search.best_params_["model__alpha"]
    best_cv_rmse = -search.best_score_
    coef = best_model.named_steps["model"].coef_
    nonzero_count = int(np.sum(np.abs(coef) > 1e-8))

    if model_name == "Ridge":
        usage_note = "更偏稳定预测 / 抗共线性"
    elif model_name == "Lasso":
        usage_note = "更偏变量筛选 / 稀疏结果"
    else:
        usage_note = "相关变量成组时更稳健"

    summary_rows.append(
        {
            "model": model_name,
            "best_alpha": round(float(best_alpha), 4),
            "best_cv_rmse": round(float(best_cv_rmse), 4),
            "nonzero_coef": nonzero_count,
            "how_to_read": usage_note,
        }
    )

summary_table = (
    pd.DataFrame(summary_rows).sort_values("best_cv_rmse").reset_index(drop=True)
)
summary_table.style.hide(axis="index")

# %% [markdown] tags=["takeaway"]
# 这一幕要收住两个判断：
#
# - `alpha` 不能靠肉眼挑，而要通过交叉验证来选；
# - 最稀疏不一定最好，最强惩罚也不一定最好。

# %% [markdown] tags=["script", "teacher-only"]
# 这幕的语气要“降温”：
# - 前面已经看了很多漂亮路径图；
# - 这里要提醒大家，最后真正决定模型选择的，是验证表现和任务目标。

# %% [markdown]
# ---
# ## 第六幕：变量选择
#
# 到这里，一个自然的问题会冒出来：
# **如果我就是想要一份“变量名单”，除了 Lasso 这种正则化方法，还有没有更经典、更直接的选法？**
#
# 这一幕专门引入：
# - 前向选择（forward selection）
# - 后向剔除（backward elimination）
# - 双向逐步回归（stepwise selection）
#
# 它们不是本周的主角，但这里需要知道：
# **变量选择并不只有 Lasso 一条路。**

# %% [markdown] tags=["prompt"]
# 如果老板说：“我不要 20 个变量，我只要 4 个。”
#
# 你觉得更自然的做法是哪一种？
#
# 1. 一个一个加进去，看谁能提升评价表现；
# 2. 从全模型开始，一个一个删掉表现最不关键的变量；
# 3. 直接让 Lasso 自动压成 0。
#
# 这三种思路，你觉得分别会有什么优点和代价？

# %% [markdown] tags=["stage"]
# **本幕主图 1**：前向选择 vs 后向剔除 的变量进入/退出路径
#
# 这张图要回答：
# > 如果目标就是得到一份更短的变量名单，前向/后向选择是怎么工作的？
#
# 课堂重点：
# - 前向选择：从空模型出发，一次加入一个最有帮助的变量；
# - 后向剔除：从全模型出发，一次删除一个最不关键的变量；
# - 两条路径可能不同，最后名单也可能不同。
#
# 先看“路径”，不要急着背定义。


# %%
def evaluate_feature_subset(X_df, y_vec, selected_features):
    if len(selected_features) == 0:
        baseline = np.full_like(y_vec, y_vec.mean(), dtype=float)
        return rmse(y_vec, baseline)

    model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    model.fit(X_df[selected_features], y_vec)
    preds = model.predict(X_df[selected_features])
    return rmse(y_vec, preds)


X_select, y_select = make_correlated_regression_data(
    n_samples=220, noise_std=1.0, seed=RANDOM_SEED
)
X_select_df = pd.DataFrame(X_select, columns=FEATURE_NAMES)

remaining = FEATURE_NAMES.copy()
forward_selected = []
forward_records = []
for step in range(len(FEATURE_NAMES)):
    candidates = []
    for feature in remaining:
        trial_features = forward_selected + [feature]
        score = evaluate_feature_subset(X_select_df, y_select, trial_features)
        candidates.append((feature, score))
    best_feature, best_score = min(candidates, key=lambda x: x[1])
    forward_selected.append(best_feature)
    remaining.remove(best_feature)
    forward_records.append(
        {
            "step": step + 1,
            "feature": best_feature,
            "rmse": best_score,
            "direction": "Forward",
        }
    )

backward_selected = FEATURE_NAMES.copy()
backward_records = []
for step in range(len(FEATURE_NAMES) - 1):
    candidates = []
    for feature in backward_selected:
        trial_features = [f for f in backward_selected if f != feature]
        score = evaluate_feature_subset(X_select_df, y_select, trial_features)
        candidates.append((feature, score, trial_features))
    removed_feature, best_score, best_features = min(candidates, key=lambda x: x[1])
    backward_selected = best_features
    backward_records.append(
        {
            "step": step + 1,
            "feature": removed_feature,
            "rmse": best_score,
            "direction": "Backward",
        }
    )

forward_df = pd.DataFrame(forward_records)
backward_df = pd.DataFrame(backward_records)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
axes[0].plot(
    forward_df["step"], forward_df["rmse"], marker="o", color="#2563eb", linewidth=2.5
)
for _, row in forward_df.iterrows():
    axes[0].annotate(
        f"+ {row['feature']}",
        (row["step"], row["rmse"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.18", fc="#dbeafe", ec="none", alpha=0.9),
    )
axes[0].set_title("Forward selection：从空模型出发，一步步加变量")
axes[0].set_xlabel("step")
axes[0].set_ylabel("training RMSE")
axes[0].invert_yaxis()

axes[1].plot(
    backward_df["step"], backward_df["rmse"], marker="o", color="#dc2626", linewidth=2.5
)
for _, row in backward_df.iterrows():
    axes[1].annotate(
        f"- {row['feature']}",
        (row["step"], row["rmse"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.18", fc="#fee2e2", ec="none", alpha=0.9),
    )
axes[1].set_title("Backward elimination：从全模型出发，一步步删变量")
axes[1].set_xlabel("step")
axes[1].invert_yaxis()

fig.suptitle("同样是“变量选择”，前向与后向会走出不同搜索路径", y=1.03, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown] tags=["stage"]
# **本幕主图 2**：Lasso / 前向选择 / 后向剔除 的最终变量名单对比
#
# 这张图要回答：
# > 不同变量选择方法，最后给出的名单会一样吗？
#
# 课堂重点：
# - 变量选择结果依赖方法；
# - 方法越贪心，越可能受当前路径影响；
# - 稀疏结果很有吸引力，但它并不是唯一答案。


# %%
def top_k_forward_selection(X_df, y_vec, k):
    remaining = list(X_df.columns)
    selected = []
    for _ in range(k):
        best_feature = None
        best_score = np.inf
        for feature in remaining:
            trial_features = selected + [feature]
            score = evaluate_feature_subset(X_df, y_vec, trial_features)
            if score < best_score:
                best_score = score
                best_feature = feature
        selected.append(best_feature)
        remaining.remove(best_feature)
    return selected


def top_k_backward_selection(X_df, y_vec, k):
    selected = list(X_df.columns)
    while len(selected) > k:
        best_features = None
        best_score = np.inf
        for feature in selected:
            trial_features = [f for f in selected if f != feature]
            score = evaluate_feature_subset(X_df, y_vec, trial_features)
            if score < best_score:
                best_score = score
                best_features = trial_features
        selected = best_features
    return selected


lasso_best = search_results["Lasso"].best_estimator_
lasso_nonzero = [
    FEATURE_NAMES[i]
    for i, coef in enumerate(lasso_best.named_steps["model"].coef_)
    if abs(coef) > 1e-8
]
forward_top4 = top_k_forward_selection(X_select_df, y_select, k=4)
backward_top4 = top_k_backward_selection(X_select_df, y_select, k=4)

selection_matrix = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "Lasso": [1 if f in lasso_nonzero else 0 for f in FEATURE_NAMES],
        "Forward": [1 if f in forward_top4 else 0 for f in FEATURE_NAMES],
        "Backward": [1 if f in backward_top4 else 0 for f in FEATURE_NAMES],
    }
)

fig, ax = plt.subplots(figsize=(11.2, 5.8))
heatmap_data = selection_matrix[["Lasso", "Forward", "Backward"]].to_numpy()
im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["Lasso", "Forward", "Backward"], fontsize=11)
ax.set_yticks(np.arange(len(FEATURE_NAMES)))
ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
for i in range(len(FEATURE_NAMES)):
    for j in range(3):
        symbol = "✓" if heatmap_data[i, j] == 1 else "·"
        ax.text(
            j,
            i,
            symbol,
            ha="center",
            va="center",
            color="black",
            fontsize=14,
            fontweight="bold",
        )
for row in range(len(FEATURE_NAMES) + 1):
    ax.axhline(row - 0.5, color="white", linewidth=1.2)
for col in range(4):
    ax.axvline(col - 0.5, color="white", linewidth=1.2)
ax.set_title("不同变量选择方法，最后留下的变量名单可能并不相同")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 变量选择方法怎么做？看什么标准？优点和代价是什么？
#
# **标准看什么？**
# - 在今天这个 notebook 里，为了把机制讲清，我们用的是误差表现（这里用 RMSE）来比较；
# - 在更完整的建模流程里，也可以看交叉验证误差、AIC/BIC、调整后 `R^2`，或者结合业务可解释性要求；
# - 关键不是死记某一个指标，而是先明确：你是在追求预测表现，还是追求更短、更可讲的变量名单。
#
# **前向选择（Forward Selection）**
# - **怎么做**：从空模型开始，每一步加入一个最能改善评价指标的变量；
# - **优点**：直观、容易讲清“变量是怎么一步步进来的”；当变量很多时，通常比穷举更省；
# - **劣势**：一旦前面加错，后面路径就会被带偏；已经加入的变量通常不容易被重新审视。
#
# **后向剔除（Backward Elimination）**
# - **怎么做**：从全模型开始，每一步删掉一个最不伤害评价指标的变量；
# - **优点**：一开始看的是“完整信息”，适合在变量数不太大时做精简；
# - **劣势**：变量很多时计算成本会高；如果变量间高度相关，删谁留谁可能会很敏感。
#
# **双向逐步回归（Stepwise Selection）**
# - **怎么做**：可以看作“前向 + 后向”的折中版；
# - **初始状态**：通常从空模型开始（也可以从一个较小的初始模型开始）；
# - **迭代状态**：每一轮先尝试加入一个最有帮助的新变量，再检查当前模型里是否有变量应该被删掉；
# - **终止状态**：当继续加入或删除都不能再明显改善评价指标时停止；
# - **优点**：比纯前向更灵活，因为“加进来以后也允许后悔”；
# - **劣势**：本质上仍然是逐步搜索，所以仍受评价标准、变量相关性和路径依赖影响，结果也不一定稳定。

# %% [markdown] tags=["takeaway"] jupyter={"source_hidden": true}
# 这一幕要收住四个判断：
#
# - 变量选择不只有 `Lasso`，还可以有前向选择、后向剔除等经典做法；
# - 前向/后向/双向方法的优点是直观、名单感强，适合讲“怎么一步步选”；
# - 它们的代价是路径依赖、计算代价上升，而且结果可能不稳定；
# - 所以变量选择是“建模策略选择”，不是自动抽取真理。

# %% [markdown] tags=["script", "teacher-only"] jupyter={"source_hidden": true}
# 这一幕不要把重点放在算法细节推导上。
# 更重要的是把方法地图立起来：
# - Lasso：把筛选写进优化问题；
# - Forward / Backward：把筛选写成逐步搜索过程；
# - 三者都能给名单，但名单不等于因果真相。
# %% [markdown]
# ## Synthesis 2
#
# 到这里，可以把三种方法压缩成一个判断框架：
#
# - 想要稳定收缩、抗共线性：想到 `Ridge`；
# - 想要稀疏结果、自动筛选：想到 `Lasso`；
# - 遇到相关变量成组、又想筛选又想稳：想到 `Elastic Net`；
# - 最后到底选哪个，不靠直觉，而要回到交叉验证结果；
# - 而即便变量被选中了，也不自动等于“它拥有因果解释地位”。

# %% [markdown]
# ## Transition
#
# 现在我们已经知道：
#
# - 模型不能只看训练集拟合；
# - penalty 会改变模型性格；
# - 最终还要靠验证结果来选模型。
#
# 那下一步，一个模型到底该怎样被更系统地比较？
#
# 这会自然引向后续更系统的模型评估与模型选择问题。
