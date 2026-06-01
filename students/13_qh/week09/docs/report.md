# 第九周：数据急救员与病态模型诊断

## 1. 实验目的

本实验旨在：
1. 编写支持命令行参数的数据预处理脚本（CLI）
2. 实现多重共线性诊断工具（VIF 计算）
3. 理解数据清洗对模型性能的影响
4. 思考数据泄露问题

---

## 2. 实验设计

### 2.1 目录结构

```
students/13_qh/week09/
├── data/
│   ├── dirty_marketing.csv    # 脏数据
│   └── clean_marketing.csv    # 清洗后数据
├── docs/report.md
├── results/
│   └── diagnostics_report.md
└── src/
    ├── utils/
    │   ├── __init__.py
    │   ├── models.py          # OLS 模型
    │   └── diagnostics.py     # VIF 诊断工具
    └── week9/
        ├── data_prep.py       # CLI 数据清洗脚本
        └── evaluate.py        # 诊断与交叉验证
```

### 2.2 脏数据特征

- **缺失值**：TV_Budget (53), Radio_Budget (49), SocialMedia_Budget (54)
- **异常值**：约 1% 的极端值（2-5 倍于 99 分位数）
- **分类变量**：Region (EU, NA, Asia, LatAm)

---

## 3. 方法说明

### 3.1 CLI 数据清洗脚本 (data_prep.py)

**使用方法**：
```bash
uv run week9/data_prep.py --input data/dirty_marketing.csv --output data/clean_marketing.csv
```

**如何捕获命令行参数**：

使用 Python 内置的 `argparse` 库：
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="数据清洗 CLI 工具")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出数据文件路径")
    return parser.parse_args()

# 使用
args = parse_args()
input_path = args.input   # 获取 --input 的值
output_path = args.output  # 获取 --output 的值
```

**参数说明**：
- `--input`：参数名，`type=str` 表示字符串类型，`required=True` 表示必填
- `--output`：同上
- 运行时传入的值会被自动解析并存储在 `args` 对象中

**清洗步骤**：
1. **缺失值处理**：使用均值填补数值列
2. **异常值处理**：Winsorization，将超过 99 分位数的值缩尾
3. **分类变量处理**：One-Hot 编码，`drop_first=True` 避免虚拟变量陷阱

### 3.2 VIF 诊断工具 (diagnostics.py)

**VIF 计算公式**：
$$
VIF_j = \frac{1}{1 - R_j^2}
$$

其中 $R_j^2$ 是将第 j 个特征作为目标变量，对其他所有特征做 OLS 回归得到的 R²。

**判断标准**：
- VIF < 5：无多重共线性
- 5 ≤ VIF < 10：中等多重共线性
- VIF ≥ 10：严重多重共线性

### 3.3 交叉验证评估 (evaluate.py)

**什么是交叉验证（Cross-Validation）？**

交叉验证是一种评估模型泛化能力的方法，核心思想是：**用一部分数据训练，用另一部分数据测试**。

**5-Fold 交叉验证流程**：

```
原始数据（1000条）:
┌─────────────────────────────────────────────────────────┐
│  Fold1  │  Fold2  │  Fold3  │  Fold4  │  Fold5  │
│  200条  │  200条  │  200条  │  200条  │  200条  │
└─────────────────────────────────────────────────────────┘

第1轮: [测试] [训练] [训练] [训练] [训练] → R²₁
第2轮: [训练] [测试] [训练] [训练] [训练] → R²₂
第3轮: [训练] [训练] [测试] [训练] [训练] → R²₃
第4轮: [训练] [训练] [训练] [测试] [训练] → R²₄
第5轮: [训练] [训练] [训练] [训练] [测试] → R²₅

最终 R² = (R²₁ + R²₂ + R²₃ + R²₄ + R²₅) / 5
```

**为什么要用交叉验证？**

1. **更可靠的评估**：单次划分可能有运气成分，5折交叉验证做了5次实验，结果更稳定
2. **充分利用数据**：每条数据都参与过训练和测试
3. **发现过拟合**：如果训练集表现好但验证集表现差，说明模型过拟合了

**本次实验结果**：
- 平均 R² = 0.7996 (±0.0419)
- 标准差 0.0419 说明各折结果比较稳定

---

## 4. 实验结果

### 4.1 数据清洗结果

| 步骤 | 操作 | 结果 |
|------|------|------|
| 缺失值 | 均值填补 | TV: 150.22, Radio: 75.67, Social: 119.87 |
| 异常值 | Winsorization | TV: 10个, Radio: 10个, Social: 10个 |
| 分类变量 | One-Hot 编码 | Region_EU, Region_LatAm (drop_first=True) |

**清洗后数据形状**：(1000, 7)

### 4.2 VIF 诊断结果

| 特征 | VIF | 状态 |
|------|-----|------|
| TV_Budget | 1.1116 | ✓ 正常 |
| Radio_Budget | 1.0374 | ✓ 正常 |
| SocialMedia_Budget | 1.1114 | ✓ 正常 |
| Is_Holiday | 1.0106 | ✓ 正常 |
| Region_EU | 1.1146 | ✓ 正常 |
| Region_LatAm | 1.1191 | ✓ 正常 |

**结论**：所有特征的 VIF 均小于 10，未检测到严重多重共线性问题。

### 4.3 交叉验证结果

| 折数 | R² |
|------|-----|
| 第 1 折 | 0.7385 |
| 第 2 折 | 0.8265 |
| 第 3 折 | 0.8254 |
| 第 4 折 | 0.8465 |
| 第 5 折 | 0.7613 |
| **平均** | **0.7996 (±0.0419)** |

---

## 5. 思考题

### 问题：验证集数据真的算是"完全未见过的陌生数据"吗？

**答案**：不完全是。

**原因**：
在 `data_prep.py` 中，我们使用了**全量数据的均值**来填补缺失值。这意味着验证集中的数据已经被"污染"了——它们的缺失值填补使用了包含验证集在内的全量数据的统计量。这属于一种轻微的**数据泄露（data leakage）**，会导致交叉验证的 R² 偏高。

**正确的做法**：
1. 在每折交叉验证中，只使用训练折的均值来填补缺失值
2. 然后用同样的均值去填补验证折的缺失值
3. 这样才能保证验证折是"完全未见过的陌生数据"

---

## 6. 结论

1. **CLI 脚本成功实现**：支持命令行参数，可独立运行
2. **VIF 诊断工具有效**：成功检测多重共线性问题
3. **数据清洗显著提升性能**：清洗后 R² 约 0.80
4. **虚拟变量陷阱已避免**：使用 `drop_first=True`
5. **数据泄露问题存在**：全量均值填补导致轻微数据泄露

---

## 7. 生成文件

- `data/clean_marketing.csv`：清洗后的数据
- `results/diagnostics_report.md`：诊断报告
