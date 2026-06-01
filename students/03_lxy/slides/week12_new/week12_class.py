# slides/week12_new/week12_class.py
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎭 Week 12: The Bias-Variance Visual Lab
# 
# ## 教学目标
# 1. 理解模型复杂度与偏差-方差的权衡
# 2. 识别欠拟合与过拟合的可视化表现
# 3. 理解 RMSE 与 MAE 的不同性格

# %% [markdown]
# ## 📦 导入必要的库

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 设置随机种子保证可复现
np.random.seed(42)

# 设置 matplotlib 样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# %% [markdown]
# ## 🎯 Task A: 构造"会过拟合"的可视化舞台
# 
# ### 问题：
# 模型复杂度增加时，为什么训练误差下降而测试误差不一定下降？

# %% [markdown]
# ### A1. 生成模拟回归数据

# %%
def generate_data(n_samples=200, noise_std=0.2, test_size=0.3):
    """生成模拟回归数据：y = sin(2πx) + 0.5x + noise"""
    X = np.random.uniform(0, 2, n_samples)
    X_sorted = np.linspace(0, 2, 1000)
    
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * x
    
    y_true = true_function(X)
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_sorted': X_sorted,
        'y_true_sorted': true_function(X_sorted)
    }

data = generate_data()
print(f"训练集: {len(data['X_train'])} 样本")
print(f"测试集: {len(data['X_test'])} 样本")

# %% [markdown]
# ### 💡 思考
# 为什么我们的真实函数要设计成 `sin(2πx) + 0.5x`？

# %% [markdown]
# ### A2. 比较三位候选模型

# %%
def train_polynomial_model(degree, X_train, y_train):
    """训练多项式回归模型"""
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train.reshape(-1, 1), y_train)
    return model

def calculate_rmse(y_true, y_pred):
    """计算 RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

degrees = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, degree in enumerate(degrees):
    # 训练模型
    model = train_polynomial_model(degree, data['X_train'], data['y_train'])
    
    # 预测
    y_train_pred = model.predict(data['X_train'].reshape(-1, 1))
    y_test_pred = model.predict(data['X_test'].reshape(-1, 1))
    
    # 计算误差
    train_rmse = calculate_rmse(data['y_train'], y_train_pred)
    test_rmse = calculate_rmse(data['y_test'], y_test_pred)
    
    # 绘制
    X_plot = np.linspace(0, 2, 500)
    y_plot_pred = model.predict(X_plot.reshape(-1, 1))
    
    ax = axes[idx]
    ax.scatter(data['X_train'], data['y_train'], alpha=0.6, s=20, 
              color='blue', label='Training')
    ax.scatter(data['X_test'], data['y_test'], alpha=0.6, s=20, 
              color='orange', label='Test')
    ax.plot(data['X_sorted'], data['y_true_sorted'], 'k--', 
           label='True', linewidth=2)
    ax.plot(X_plot, y_plot_pred, color='red', linewidth=2, 
           label=f'Degree={degree}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}\nTrain RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/candidate_models.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 📊 课堂讨论
# 
# 1. **谁最像欠拟合？** Degree=1 的线性模型
# 2. **谁最像过拟合？** Degree=15 的高阶多项式
# 3. **如果必须选一个上线，你会先押谁？** Degree=4，因为它泛化最好

# %% [markdown]
# ## 📈 Task B: 画出完整的复杂度-误差曲线

# %%
degrees = range(1, 19)
train_errors = []
test_errors = []

print("Degree | Train RMSE | Test RMSE | Gap")
print("-" * 45)

for degree in degrees:
    model = train_polynomial_model(degree, data['X_train'], data['y_train'])
    
    y_train_pred = model.predict(data['X_train'].reshape(-1, 1))
    y_test_pred = model.predict(data['X_test'].reshape(-1, 1))
    
    train_rmse = calculate_rmse(data['y_train'], y_train_pred)
    test_rmse = calculate_rmse(data['y_test'], y_test_pred)
    gap = test_rmse - train_rmse
    
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    if degree % 3 == 0:
        print(f"{degree:3d}   | {train_rmse:9.4f} | {test_rmse:8.4f} | {gap:8.4f}")

# %%
# 绘制误差曲线
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(degrees, train_errors, 'o-', label='Train RMSE', 
       linewidth=2, markersize=6, color='blue')
ax.plot(degrees, test_errors, 's-', label='Test RMSE', 
       linewidth=2, markersize=6, color='red')

best_degree = degrees[np.argmin(test_errors)]
best_error = min(test_errors)
ax.axvline(x=best_degree, color='green', linestyle='--', alpha=0.5, 
          label=f'Best: degree={best_degree}')
ax.scatter(best_degree, best_error, color='green', s=100, zorder=5)

ax.set_xlabel('Model Complexity (Polynomial Degree)')
ax.set_ylabel('RMSE')
ax.set_title('Model Complexity vs Error Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/error_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ 最优复杂度: degree={best_degree}, Test RMSE={best_error:.4f}")

# %% [markdown]
# ### 💡 关键洞察
# 
# 训练误差持续下降，但测试误差先降后升 → **过拟合**

# %% [markdown]
# ## 🌪 Task C: 用 repeated sampling 把 variance 画出来

# %%
def variance_demo(degree, n_repeats=20):
    """演示特定复杂度的方差"""
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * x
    
    all_predictions = []
    
    for repeat in range(n_repeats):
        # 重新生成数据
        X = np.random.uniform(0, 2, 200)
        y_true = true_function(X)
        noise = np.random.normal(0, 0.2, 200)
        y = y_true + noise
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=repeat)
        
        model = train_polynomial_model(degree, X_train, y_train)
        
        X_plot = np.linspace(0, 2, 200)
        y_pred = model.predict(X_plot.reshape(-1, 1))
        all_predictions.append(y_pred)
    
    return np.array(all_predictions)

# %%
# 对比 low variance vs high variance
degrees_to_compare = [2, 15]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

def true_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * x

for idx, degree in enumerate(degrees_to_compare):
    all_preds = variance_demo(degree, n_repeats=20)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    X_plot = np.linspace(0, 2, 200)
    y_true_plot = true_function(X_plot)
    
    ax = axes[idx]
    # 绘制所有预测曲线（半透明）
    for pred in all_preds[:10]:
        ax.plot(X_plot, pred, 'b-', alpha=0.2, linewidth=1)
    
    ax.plot(X_plot, y_true_plot, 'k-', linewidth=3, label='True')
    ax.fill_between(X_plot, mean_pred - std_pred, mean_pred + std_pred, 
                   alpha=0.3, color='red', label='±1 Std')
    ax.plot(X_plot, mean_pred, 'r--', linewidth=2, label='Mean')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}\nMean Std={np.mean(std_pred):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/variance_demo.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 🎯 课堂投票
# 
# **high variance model 的危险，不是它不会拟合训练集，而是它对 ______ 过于敏感。**
# 
# 答案：**训练样本的随机波动**

# %% [markdown]
# ## 💥 Task D: 让异常值攻击 RMSE 与 MAE

# %%
# 构造对比实验
n_points = 100
np.random.seed(42)
y_true = np.random.normal(10, 2, n_points)
y_pred_clean = y_true + np.random.normal(0, 0.5, n_points)

# 人为加入一个明显离群点
y_pred_outlier = y_pred_clean.copy()
outlier_idx = 5
y_pred_outlier[outlier_idx] = y_true[outlier_idx] + 20

# 计算指标
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

rmse_clean = calculate_rmse(y_true, y_pred_clean)
mae_clean = calculate_mae(y_true, y_pred_clean)
rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
mae_outlier = calculate_mae(y_true, y_pred_outlier)

print("=" * 50)
print("RMSE vs MAE 对比实验")
print("=" * 50)
print(f"干净场景:  RMSE={rmse_clean:.4f}, MAE={mae_clean:.4f}")
print(f"含异常值:  RMSE={rmse_outlier:.4f}, MAE={mae_outlier:.4f}")
print(f"变化率:    RMSE +{(rmse_outlier-rmse_clean)/rmse_clean*100:.1f}%, "
      f"MAE +{(mae_outlier-mae_clean)/mae_clean*100:.1f}%")

# %%
# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左侧：误差分布
errors_clean = y_pred_clean - y_true
errors_outlier = y_pred_outlier - y_true

ax1 = axes[0]
ax1.boxplot([errors_clean, errors_outlier], labels=['Clean', 'With Outlier'], widths=0.6)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax1.set_ylabel('Prediction Error')
ax1.set_title('Error Distribution Comparison')
ax1.grid(True, alpha=0.3)

# 右侧：指标对比
ax2 = axes[1]
x_pos = np.arange(2)
width = 0.35

rmse_bars = ax2.bar(x_pos - width/2, [rmse_clean, rmse_outlier], 
                   width, label='RMSE', color='blue', alpha=0.7)
mae_bars = ax2.bar(x_pos + width/2, [mae_clean, mae_outlier], 
                  width, label='MAE', color='orange', alpha=0.7)

for bar in rmse_bars + mae_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Clean', 'With Outlier'])
ax2.set_ylabel('Error')
ax2.set_title('RMSE is More Sensitive to Outliers')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/loss_outlier_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 📝 业务解释
# 
# | 场景 | 推荐指标 | 原因 |
# |------|---------|------|
# | 大错误代价极高（医疗、金融） | **RMSE** | 惩罚极端错误 |
# | 数据含较多异常值 | **MAE** | 对异常值不敏感 |

# %% [markdown]
# ## 🔗 与下一周的联系
# 
# > 如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？
# 
# **答案**：
# - 正则化通过在损失函数中添加惩罚项来控制模型复杂度
# - Ridge (L2)：让系数趋近于零但非零
# - Lasso (L1)：可将部分系数压缩为零（特征选择）
# - 目标：**在不增加模型容量的前提下，提升泛化能力**

# %% [markdown]
# ## 📊 总结：本章最重要的一张图
# 
# **`candidate_models.png` 中 degree=15 的子图**最能代表过拟合：
# - ✅ 训练点几乎完美拟合（Train RMSE 极低）
# - ❌ 测试点误差很大（Test RMSE 显著升高）
# - 📈 预测曲线剧烈震荡，在训练点之间来回摆动

# %% [markdown]
# ---
# 
# ## 🏠 课后作业
# 
# 请完成以下任务并提交到 `students/03_1xy/src/week12/`：
# 
# 1. **main.py**：可独立运行的脚本
# 2. **results/summary.md**：分析报告
# 3. **results/figures/**：所有生成的图表
# 
# ### 必答问题
# 
# 1. 模型复杂度增加时，为什么训练误差下降而测试误差不一定下降？
# 2. 什么叫 high variance，它在图上长什么样？
# 3. 为什么异常值会让 RMSE 和 MAE 表现出不同的性格？