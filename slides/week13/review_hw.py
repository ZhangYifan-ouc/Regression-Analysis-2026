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
# # Week 13：Week 12 作业答辩 Review（源码机制审计版）
#
# 这一版 notebook 不急着“修图”，而是先做一件更基础的事：
#
# **把 5 位同学源码里的数据生成、特征构造、随机数状态、split 机制、求解器、报告生成链路逐项查清。**
#
# 目标不是快速下结论，而是避免把不同类型的问题混为一谈。

# %% [markdown] tags=["script", "teacher-only"]
# **Teacher note**
# - 这版重点不是“展示一个漂亮修复图”，而是训练学生如何从源码细节定位根因。
# - 每位同学都先回答：`X` 怎么来、噪声怎么来、split 怎么来、模型怎么拟合、报告怎么写。
# - 先做机制审计，再谈如何修复。

# %% [markdown] tags=["prompt"]
# ## 00. 先定一个“彻查模板”
#
# 对每位同学，都不要先说“这是过拟合”或“这是 split 运气”。
#
# 先按这 6 项逐条查：
#
# 1. **`X` 的生成方式**：`linspace` 还是 `uniform`？区间多大？
# 2. **噪声机制**：噪声来自全局 `np.random`，还是局部 `default_rng(seed)`？
# 3. **split 机制**：单次 split 还是重复多次？split 是否和噪声复用同一随机源？
# 4. **特征矩阵**：是否手写多项式特征？是否标准化？是否 include bias？
# 5. **求解器**：是纯 OLS、`LinearRegression`、还是带正则的模型？
# 6. **报告链路**：报告是自动生成，还是手写？是否存在图表和文字脱节风险？

# %% [markdown] tags=["takeaway"]
# **课堂锚点**
#
# 同样一张“奇怪的图”，可能对应完全不同的根因：
# - 有的是 **病态设计矩阵**；
# - 有的是 **单次 split 的偶然性**；
# - 有的是 **全局随机状态推进** 导致的不可对齐；
# - 有的是 **手写报告** 没有核对结果。
#
# 不做源码机制审计，就很容易误判。

# %% [markdown] tags=["prompt"]
# ## 01. 12_wxy：危险的不只是 `[0,10]`，而是 `[0,10]` 进入了手写高次幂矩阵
#
# 对 `12_wxy`，我们要具体问：
#
# - `X` 进入 `polynomial_features` 后会变成什么矩阵？
# - 每一列的量级怎么变化？
# - 截距、特征、求解器是怎么拼在一起的？

# %% [markdown] tags=["stage"]
# ![12_wxy original](figures/12_wxy_error_curves.png)

# %% [markdown] tags=["stage"]
# ### 12_wxy：数据生成 + 手写特征 + 求解器（摘录）
#
# ```python
# np.random.seed(42)
#
# def polynomial_features(X, degree):
#     n_samples = X.shape[0]
#     features = [np.ones(n_samples)]
#     for d in range(1, degree + 1):
#         features.append(X.reshape(-1) ** d)
#     return np.column_stack(features[1:])
#
# def generate_data(n_samples=150):
#     x = np.linspace(0, 10, n_samples).reshape(-1, 1)
#     y_true = np.sin(x).ravel()
#     y = y_true + np.random.normal(0, 0.3, size=n_samples)
#     return x, y, y_true
#
# def train_test_split(x, y, test_ratio=0.3):
#     indices = np.random.permutation(len(x))
#     test_size = int(len(x) * test_ratio)
#     test_idx = indices[:test_size]
#     train_idx = indices[test_size:]
#     return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
#
# model = CustomOLS(fit_intercept=True, alpha=0.0)
# ```

# %% [markdown] tags=["stage"]
# ### 12_wxy：源码机制审计
#
# - `X` 生成方式：`np.linspace(0, 10, n_samples)`，不是随机抽样，而是规则均匀铺点。
# - 噪声机制：`np.random.normal(...)`，依赖全局 `np.random` 状态。
# - split 机制：自写 `np.random.permutation(...)`，同样依赖全局 `np.random` 状态。
# - 特征矩阵：`polynomial_features` 手写生成 `[x, x^2, ..., x^d]`，没有标准化。
# - 一个关键细节：函数里先创建 `np.ones`，但返回时切掉了常数列，说明截距不是靠特征矩阵，而是交给 `CustomOLS(fit_intercept=True)`。
# - 求解器：`CustomOLS(alpha=0.0)`，等价于无正则纯 OLS。
# - 风险组合：`x ∈ [0,10]` 进入高次幂后，列量级从 `10^1` 一路涨到 `10^18`；列间又高度相关，极易形成病态矩阵。
# - 真正危险的不是单独的 `[0,10]`，而是 `[0,10]` 被送入手写高次幂设计矩阵，再用无正则 OLS 去解。
# - 因此图上的 train 锯齿不是“普通过拟合”，而更像数值求解已经失稳。
# - 另一个次要问题：Task A / B 各自独立重新生成数据，因此不同任务之间的图不能严格互相印证。

# %% [markdown] tags=["takeaway"]
# **这位同学最该补的细节**
#
# `12_wxy` 的主问题不是“X 不是随机所以不好”，而是：
#
# > `linspace(0,10)` 被送进手写 `x, x^2, ..., x^d` 设计矩阵后，矩阵列尺度爆炸、列间高度相关，而求解器又是无正则 OLS。
#
# 这才是 train 曲线锯齿的根因链条。

# %% [markdown] tags=["prompt"]
# ## 02. 21_yyw：真正该查的是“随机 X + 单次 split”到底怎么绑定在一起的
#
# 对 `21_yyw`，重点不是病态矩阵，而是：
#
# - `X`、噪声、split 是否都绑在同一个 `seed=42` 上？
# - 这会不会把一次偶然划分固化成最终课堂图？

# %% [markdown] tags=["stage"]
# ![21_yyw original](figures/21_yyw_error_curves.png)

# %% [markdown] tags=["stage"]
# ### 21_yyw：数据生成 + split + 模型（摘录）
#
# ```python
# RANDOM_STATE = 42
# N_SAMPLES = 200
# NOISE_STD = 0.5
# TEST_RATIO = 0.3
#
# def make_poly_model(degree: int) -> Pipeline:
#     return Pipeline(
#         [
#             ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
#             ("lr", LinearRegression()),
#         ]
#     )
#
# def generate_data(
#     n_samples: int = N_SAMPLES,
#     noise_std: float = NOISE_STD,
#     test_ratio: float = TEST_RATIO,
#     seed: int = RANDOM_STATE,
# ):
#     rng = np.random.default_rng(seed)
#     x = rng.uniform(-3, 3, n_samples)
#     y_true = true_function(x)
#     y_noisy = y_true + rng.normal(0, noise_std, n_samples)
#
#     n_test = int(n_samples * test_ratio)
#     indices = rng.permutation(n_samples)
#     test_idx = indices[:n_test]
#     train_idx = indices[n_test:]
# ```

# %% [markdown] tags=["stage"]
# ### 21_yyw：源码机制审计
#
# - `X` 生成方式：`rng.uniform(-3, 3, n_samples)`，是真正的随机抽样。
# - 噪声机制：`rng.normal(...)`，与 `X` 由同一个局部 `rng` 控制。
# - split 机制：`rng.permutation(...)`，仍然使用同一个 `rng`。这意味着“抽样 + 加噪 + 切分”被同一个 seed 一次性固定。
# - 这个机制的后果是：`seed=42` 不只是固定了一次 split，而是把整个数据宇宙都冻结成了一个具体 realization。
# - 特征矩阵：sklearn `PolynomialFeatures(include_bias=False)`，没有手写特征矩阵。
# - 求解器：`LinearRegression()`，没有正则，但在 `x ∈ [-3,3]` 的区间下数值风险远弱于 `[0,10]`。
# - 为什么高阶还比较稳定：因为这里没有把 `[0,10]` 的大区间高次幂直接喂进病态 OLS。
# - 主问题：单次 realization 恰好让 test 集比 train 集更容易拟合，导致 `test RMSE < train RMSE` 长期出现。
# - 优点：Task A / B 共用同一份 `x_train, y_train, x_test, y_test`，任务间是一致的。
# - 缺点：一致性虽好，但如果这一次 realization 本身不典型，就会把“不典型图”稳定产出。

# %% [markdown] tags=["takeaway"]
# **这位同学最该补的细节**
#
# `21_yyw` 不是“模型写坏了”，而是：
#
# > `X`、噪声、split 都绑在同一个 `default_rng(seed=42)` 上，一次 realization 被完整冻结，结果恰好把 test 集变成了更容易拟合的一边。
#
# 所以这里的重点是 **评估机制**，不是先怀疑数值崩坏。

# %% [markdown] tags=["prompt"]
# ## 03. 18_mxt：这位的关键不是图，而是“全局随机状态 + 手写报告”
#
# 对 `18_mxt`，我们不需要反复盯图，而要查：
#
# - 为什么两个任务看起来都 `random_state=42`，却还是可能数据不一致？
# - 为什么表和文字可以彼此打架？

# %% [markdown] tags=["stage"]
# ### 18_mxt：数据生成 + 任务入口 + 报告链路（摘录）
#
# ```python
# np.random.seed(42)
#
# def generate_data(n_samples=200):
#     x = np.linspace(0, 10, n_samples)
#     y_true = np.sin(x) + 0.5 * x
#     y = y_true + np.random.normal(0, 0.3, n_samples)
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.3, random_state=42
#     )
#     return x, y_true, x_train, x_test, y_train, y_test
#
# def run_candidate_models():
#     x, y_true, x_train, x_test, y_train, y_test = generate_data()
#
# def run_complexity_sweep():
#     x, y_true, x_train, x_test, y_train, y_test = generate_data()
#
# print("⚠️  This script only runs experiments and generates plots/data")
# print("⚠️  Please write your report manually to: results/summary.md")
# ```

# %% [markdown] tags=["stage"]
# ### 18_mxt：源码机制审计
#
# - `X` 生成方式：`np.linspace(0,10, n_samples)`，不是随机抽样。
# - 噪声机制：`np.random.normal(...)`，走全局 `np.random` 状态。
# - split 机制：sklearn `train_test_split(..., random_state=42)`。表面看 split 固定，但输入 `y` 在两次 `generate_data()` 之间已经不同。
# - 关键细节：`run_candidate_models()` 和 `run_complexity_sweep()` 各自独立调用 `generate_data()`。
# - 所以虽然 `train_test_split` 的 `random_state=42` 没变，但它切分的是两份不同噪声 realization。
# - 这就造成了任务间不可直接比较：不是 split 随机了，而是上游数据已经变了。
# - 求解器：sklearn `LinearRegression()` + `PolynomialFeatures`，不像 `12_wxy` 那样有明显自写 OLS 风险。
# - 报告链路：脚本明确打印“Please write your report manually”，说明 summary 是手写，不是程序自动生成。
# - 因此核心风险不是数值稳定性，而是：任务间数据不一致 + 手写报告未核对。
# - 这解释了为什么 Task A 表格说一套，文字结论却能写成另一套。

# %% [markdown] tags=["takeaway"]
# **这位同学最该补的细节**
#
# `18_mxt` 的关键不是“X 在 `[0,10]` 上就一定坏”，而是：
#
# > 全局 `np.random` 让两次 `generate_data()` 的噪声 realization 不同，而报告又是手写，所以非常容易出现任务间不一致和文字-表格脱节。

# %% [markdown] tags=["prompt"]
# ## 04. 07_nc：这位恰好是“机制相对健康”的对照组
#
# 对 `07_nc`，要问的不是“哪里炸了”，而是：
#
# - 为什么她没有炸？
# - 现在的图和代码是否支持“标准过拟合”的叙事？

# %% [markdown] tags=["stage"]
# ### 07_nc：数据生成 + split + pipeline（摘录）
#
# ```python
# rng = np.random.default_rng(seed)
# x = np.sort(rng.uniform(-3.0, 3.0, size=N_SAMPLES))
# noise = rng.normal(0.0, 0.35, size=N_SAMPLES)
# y = true_function(x) + noise
#
# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     train_size=42,
#     random_state=seed,
#     shuffle=True,
# )
#
# def make_polynomial_model(degree: int) -> Pipeline:
#     return Pipeline(
#         steps=[
#             ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
#             ("scale", StandardScaler()),
#             ("linear", LinearRegression()),
#         ]
#     )
# ```

# %% [markdown] tags=["stage"]
# ### 07_nc：源码机制审计
#
# - `X` 生成方式：`uniform(-3,3)`，区间温和。
# - 噪声机制：`default_rng(seed)` 生成，和 `X` 同源，但局部且可控。
# - split 机制：sklearn `train_test_split(..., random_state=seed)`。
# - 训练集大小：显式只取 42 个点，这本身会增加方差，但不是数值病态。
# - 特征矩阵：sklearn `PolynomialFeatures(include_bias=False)`。
# - 关键修复：pipeline 中加入了 `StandardScaler()`。
# - 求解器：`LinearRegression()`，在温和区间 + scaler 下整体稳定。
# - 因此高阶模型虽然会过拟合，但更像正常 statistical overfitting，而不是 solver 崩坏。
# - 这位同学的重要价值在于：她证明了“高阶 + 小样本”并不一定必然炸成锯齿，只要机制更稳，图仍然能讲通。
# - 所以她更适合作为对照组：帮助我们区分“正常过拟合”和“数值不稳定”。

# %% [markdown] tags=["takeaway"]
# **这位同学最该补的细节**
#
# `07_nc` 值得强调的不是“她之前也出过问题”，而是：
#
# > 在 `uniform(-3,3)` + `StandardScaler` + sklearn pipeline 的机制下，高阶模型可以表现为标准过拟合，而不是数值崩坏。

# %% [markdown] tags=["prompt"]
# ## 05. 23_zy：这位最复杂，因为“随机 X + 大区间”把两类风险叠在了一起
#
# 对 `23_zy`，要特别细查：
#
# - 为什么她和 `21_yyw` 都是随机 `X`，但更容易出数值问题？
# - 为什么她和 `12_wxy` 都有 `[0,10]`，但又多了一层随机性风险？

# %% [markdown] tags=["stage"]
# ![23_zy original](figures/23_zy_error_curves.png)

# %% [markdown] tags=["stage"]
# ### 23_zy：数据生成 + pipeline（摘录）
#
# ```python
# def generate_data(n_samples=160, noise_std=0.35, random_state=42):
#     rng = np.random.default_rng(random_state)
#
#     x = rng.uniform(0, 10, size=n_samples)
#     x = np.sort(x)
#
#     y_clean = true_function(x)
#     y = y_clean + rng.normal(0, noise_std, size=n_samples)
#
#     x_train, x_test, y_train, y_test = train_test_split(
#         x.reshape(-1, 1),
#         y,
#         test_size=0.3,
#         random_state=random_state,
#     )
#
# def build_polynomial_model(degree):
#     model = Pipeline(
#         steps=[
#             ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
#             ("linear", LinearRegression()),
#         ]
#     )
# ```

# %% [markdown] tags=["stage"]
# ### 23_zy：源码机制审计
#
# - `X` 生成方式：`rng.uniform(0,10, size=n_samples)`，既是随机抽样，又落在大区间 `[0,10]`。
# - 噪声机制：`rng.normal(...)`，与 `X` 同一个局部 `rng`。
# - split 机制：sklearn `train_test_split(..., random_state=random_state)`，由同一个随机状态参数固定。
# - 关键区别：她不像 `12_wxy` 那样是规则均匀铺点，而是随机覆盖 `[0,10]`。这会让局部稀疏/密集分布也参与风险形成。
# - 特征矩阵：`PolynomialFeatures(include_bias=False)`，没有 `StandardScaler`。
# - 求解器：`LinearRegression()`，虽然不是自写 OLS，但在 `[0,10]` 高次幂下依然会面对巨大尺度跨度。
# - 所以她同时叠加了两类风险：一类是大区间高次幂带来的数值条件数恶化；另一类是随机抽样 + 单次 split 带来的偶然性。
# - 这解释了为什么她的图既可能出现 train 不单调，也可能出现 test 长期低于 train。
# - 和 `21_yyw` 的区别在于：`21` 的 `X ∈ [-3,3]`，而 `23` 的 `X ∈ [0,10]`，所以 `23` 多了一层高阶尺度风险。
# - 和 `12_wxy` 的区别在于：`12` 是规则铺点 + 手写 OLS；`23` 是随机抽样 + sklearn OLS，但随机覆盖本身会让异常图更难解释。

# %% [markdown] tags=["takeaway"]
# **这位同学最该补的细节**
#
# `23_zy` 不是简单的“和 12 一样”或“和 21 一样”，而是：
#
# > `uniform(0,10)` 把随机抽样风险和大区间高次幂风险叠在了一起，所以它天然比 `uniform(-3,3)` 更危险，也比 `linspace(0,10)` 更复杂。

# %% [markdown] tags=["prompt"]
# ## 06. 彻查后，有没有之前没显式汇总的问题？
#
# 有，而且不止一个。下面把容易漏掉的点单列出来。

# %% [markdown] tags=["stage"]
# ### 之前容易漏掉、现在应显式汇总的问题
#
# - **全局 `np.random` 状态推进问题**：`12_wxy` 和 `18_mxt` 都依赖全局 `np.random`，这意味着不同函数调用顺序会改变后续 realization。
# - **任务间数据一致性问题**：`12_wxy`、`18_mxt` 的多个 Task 不是明确共享一份上游数据，跨任务比较容易失真。
# - **`X` 是否规则铺点的问题**：`linspace` 与 `uniform` 的区别，之前只被粗略提到，没有被当成根因分析主轴。
# - **特征矩阵构造责任归属问题**：`12_wxy` 的病根不只是 “没标准化”，而是手写 `polynomial_features` 直接生成病态幂矩阵。
# - **截距处理位置问题**：`12_wxy` 的常数列在特征函数中被创建又删除，截距交给 `fit_intercept=True`，这说明问题集中在幂次列本身。
# - **报告链路自动化程度问题**：`18_mxt` 明确要求手写报告，这比自动生成报告更容易出现数字-文字脱节。
# - **同一个 seed 绑定多个随机阶段的问题**：`21_yyw`、`23_zy` 中，同一个 seed 同时控制 `X`、噪声和 split，造成“一次 realization 被完整冻结”。
# - **为什么有的人高阶稳定、有的人不稳定**：这不能只归结为 degree 高低，还要同时看 `X` 区间、特征尺度、是否标准化、是否手写 OLS。
# - **原图能否拿来教学的问题**：某些图不是“错”，而是“不适合作为课堂主图”；这在 `21_yyw` 特别明显。
# - **修复动作层级不同**：有的问题要修 pipeline，有的问题要修随机机制，有的问题要修报告流程，不能一律说“加正则化”。

# %% [markdown] tags=["prompt"]
# ## 07. 总结对照表：五位同学到底各自错在哪一层？

# %% tags=["stage"]
import pandas as pd

summary_df = pd.DataFrame(
    [
        [
            "12_wxy",
            "linspace(0,10)",
            "全局 np.random",
            "手写幂矩阵 + CustomOLS(alpha=0)",
            "数值病态 / solver 不稳",
            "Task 间数据不一致",
        ],
        [
            "21_yyw",
            "uniform(-3,3)",
            "default_rng(seed)",
            "PolynomialFeatures + LinearRegression",
            "单次 realization / split 运气",
            "整套数据宇宙被 seed=42 冻结",
        ],
        [
            "18_mxt",
            "linspace(0,10)",
            "全局 np.random",
            "PolynomialFeatures + LinearRegression",
            "报告未核对",
            "任务间噪声 realization 不同 + 手写报告",
        ],
        [
            "07_nc",
            "uniform(-3,3)",
            "default_rng(seed)",
            "PolynomialFeatures + StandardScaler + LinearRegression",
            "标准过拟合",
            "训练集仅 42 个点但机制整体健康",
        ],
        [
            "23_zy",
            "uniform(0,10)",
            "default_rng(random_state)",
            "PolynomialFeatures + LinearRegression",
            "随机性 + 大区间双风险",
            "单次 split 与尺度风险同时存在",
        ],
    ],
    columns=[
        "同学",
        "X 生成方式",
        "随机数机制",
        "特征/求解器",
        "主问题层级",
        "容易漏掉的次级问题",
    ],
)
summary_df

# %% [markdown] tags=["takeaway"]
# **最终收束**
#
# 这 5 位同学的问题，不应该再被笼统讲成“某某图不标准”。
#
# 更准确的说法是：
# - `12_wxy`：病态设计矩阵问题；
# - `21_yyw`：单次 realization 问题；
# - `18_mxt`：报告链路问题；
# - `07_nc`：机制健康下的标准过拟合；
# - `23_zy`：随机抽样与大区间尺度双重风险问题。
#
# 只有把这一层讲清，后面的修复才不会流于“想到什么改什么”。
