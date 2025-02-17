# Xia认知控制模型工具包

这是一个用于认知控制建模的Julia工具包，专注于贝叶斯学习和强化学习模型的实现。

### 版本
- 当前版本: 2.0.0
- 状态: 开发中

## 项目结构

```
xia_cog_model_2/
├── models/                 # 模型实现
│   └── reinforcement_learning/  # 强化学习模型
│       ├── base.jl             # 基础类型和接口
│       ├── bayesian_models.jl  # 贝叶斯学习模型
│       ├── q_learning.jl       # Q-learning模型
│       ├── stan_models.jl      # Stan模型实现
│       └── model_fitting.jl    # 模型拟合工具
├── config/                # 配置文件
├── analysis/             # 分析工具
├── utils/               # 通用工具
├── visualization/       # 可视化工具
└── tests/              # 单元测试
```

## 主要功能

### 贝叶斯学习模型
- AB测试模型
  - 基础实现
  - Stan实现（完整后验推断）
- 空间反应模型
  - 单一学习率
  - 双学习率(稳定/波动)
  - Stan实现（支持层级建模）
- 支持波动性和衰减参数

### 强化学习模型
- Q-Learning模型
  - 基础实现
  - 双学习率变体
  - Stan实现（完整后验推断）
- 状态-动作学习
- 策略提取
- 状态值计算

### Stan模型特性
- 完整的贝叶斯推断
- 参数后验分布
- 预测不确定性估计
- 模型比较指标
- 支持层级建模

## 安装

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/xia_cog_model.git")
```

### 依赖
需要安装以下依赖：
```julia
] add StanSample StatsFuns DataFrames GLM StatsBase
```

## 使用示例

### 基础模型
```julia
using xia_cog_model_2

# 创建模型
model = BayesianSpatialModel(
    learning_rate = 0.1,
    decay_rate = 0.05
)

# 拟合数据
data = load_example_data()
fit!(model, data)

# 预测
predictions = predict(model, test_data)
```

### Stan模型
```julia
# 创建Stan模型
model = StanBayesianModel(
    alpha_prior = 0.5,  # 学习率先验
    beta_prior = 0.2    # 衰减率先验
)

# 拟合数据（自动进行MCMC采样）
results = fit!(model, data)

# 获取预测及其不确定性
predictions = predict(model, test_data)
```

更多示例请参考 `examples/` 目录下的Jupyter notebooks。

## 最近更新

### 2025-02-17
- 重构项目结构为xia_cog_model_2
- 增加数据验证和错误处理
- 添加Stan模型实现
- 改进数值稳定性
- 完善代码文档

## 文档

详细文档请参考 `docs/` 目录：
- API参考
- 教程
- 示例
- 开发指南

## 贡献

欢迎提交Issue和Pull Request！

## 许可

MIT License