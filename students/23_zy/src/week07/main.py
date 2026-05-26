import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000, penalty=None, C=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.C = C
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # 交叉熵损失
            loss = (-1 / n_samples) * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
            self.loss_history.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            if self.penalty == 'l2':
                dw += (1 / self.C) * self.weights
            elif self.penalty == 'l1':
                dw += (1 / self.C) * np.sign(self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

# ---------------------- 数据加载与预处理（适配你的数据） ----------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("数据前5行：")
    print(df.head())

    # 1. 删除无关列
    if 'Region' in df.columns:
        df = df.drop(columns=['Region'])

    # 2. 把连续的Sales转成二分类标签（高于平均值为1，否则为0）
    sales_mean = df['Sales'].mean()
    print(f"\n销量平均值：{sales_mean:.2f}")
    df['Sales_High'] = (df['Sales'] > sales_mean).astype(int)
    print(f"二分类标签分布：\n{df['Sales_High'].value_counts()}")

    # 3. 特征和目标
    feature_cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
    target_col = 'Sales_High'

    X = df[feature_cols].values
    y = df[target_col].values

    # 4. 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 5. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, sales_mean

def evaluate_model(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    print("\n===== 模型评估 =====")
    print(f"准确率：{accuracy:.4f}")
    print(f"精确率：{precision:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    print("混淆矩阵：")
    print(cm)
    print(f"AUC：{roc_auc:.4f}")

    return accuracy, precision, recall, f1, cm, fpr, tpr, roc_auc

def plot_loss_curve(loss_history, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve of Gradient Descent')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    # 1. 加载数据
    DATA_PATH = "q3_marketing.csv"
    X_train, X_test, y_train, y_test, sales_mean = load_and_preprocess_data(DATA_PATH)

    # 2. 训练无正则化模型
    print("\n===== 无正则化模型 =====")
    model_none = LogisticRegressionGD(lr=0.01, n_iters=1000, penalty=None)
    model_none.fit(X_train, y_train)
    y_pred_none = model_none.predict(X_test)
    y_proba_none = model_none.predict_proba(X_test)
    acc_none, pre_none, rec_none, f1_none, _, _, _, _ = evaluate_model(y_test, y_pred_none, y_proba_none)

    # 3. 训练L2正则化模型
    print("\n===== L2正则化模型 =====")
    model_l2 = LogisticRegressionGD(lr=0.01, n_iters=1000, penalty='l2', C=1.0)
    model_l2.fit(X_train, y_train)
    y_pred_l2 = model_l2.predict(X_test)
    y_proba_l2 = model_l2.predict_proba(X_test)
    acc_l2, pre_l2, rec_l2, f1_l2, cm_l2, fpr, tpr, roc_auc = evaluate_model(y_test, y_pred_l2, y_proba_l2)

    # 4. 画图
    plot_loss_curve(model_l2.loss_history, "results/loss_curve.png")
    plot_roc_curve(fpr, tpr, roc_auc, "results/roc_curve.png")

    # 5. 生成报告
    with open("results/report.md", "w", encoding="utf-8") as f:
        f.write("# Week07 逻辑回归分析报告（23_zy）\n\n")
        f.write("## 1. 数据概况\n")
        f.write(f"- 训练集样本数：{len(X_train)}\n")
        f.write(f"- 测试集样本数：{len(X_test)}\n")
        f.write(f"- 特征数：{X_train.shape[1]}\n")
        f.write(f"- 销量平均值：{sales_mean:.2f}，以平均值为界将销量分为高/低两类\n\n")

        f.write("## 2. 模型对比结果\n")
        f.write("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |\n")
        f.write("|------|--------|--------|--------|--------|-----|\n")
        f.write(f"| 无正则化 | {acc_none:.4f} | {pre_none:.4f} | {rec_none:.4f} | {f1_none:.4f} | {auc(*roc_curve(y_test, y_proba_none)[:2]):.4f} |\n")
        f.write(f"| L2正则化 | {acc_l2:.4f} | {pre_l2:.4f} | {rec_l2:.4f} | {f1_l2:.4f} | {roc_auc:.4f} |\n")

    print("\n🎉 作业完成！报告和图片已生成在results文件夹中。")
